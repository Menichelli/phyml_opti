/*

PhyML:  a program that  computes maximum likelihood phylogenies from
DNA or AA homologous sequences.

Copyright (C) Stephane Guindon. Oct 2003 onward.

All parts of the source except where indicated are distributed under
the GNU public licence. See http://www.opensource.org for details.

*/

#include "assert.h"
#include "lk.h"
#ifdef BEAGLE
#include "beagle_utils.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#ifndef PHYML_OPT_PARTIAL_LK
#define PHYML_OPT_PARTIAL_LK 1
#endif

#ifndef PHYML_MT_LK
#define PHYML_MT_LK 1
#endif

#if PHYML_MT_LK && defined(_OPENMP)
#include <omp.h>
#define PHYML_MT_LK_RUNTIME 1
#else
#define PHYML_MT_LK_RUNTIME 0
#endif

#if PHYML_MT_LK_RUNTIME
typedef enum
{
  PHYML_MT_BACKEND_SCALAR = 0,
  PHYML_MT_BACKEND_SSE = 1,
  PHYML_MT_BACKEND_AVX = 2
} t_phyml_mt_backend;

typedef enum
{
  PHYML_MT_PHASE_WAVEFRONT = 0,
  PHYML_MT_PHASE_PARTIAL_LK = 1,
  PHYML_MT_PHASE_PMAT = 2,
  PHYML_MT_PHASE_EIGEN = 3,
  PHYML_MT_PHASE_SITE_LK = 4,
  PHYML_MT_PHASE_DLK = 5,
  PHYML_MT_PHASE_COUNT = 6
} t_phyml_mt_phase;

typedef enum
{
  PHYML_MT_UPDATE_ALL_PARTIAL_MODE_LEGACY = 0,
  PHYML_MT_UPDATE_ALL_PARTIAL_MODE_WAVEFRONT = 1
} t_phyml_mt_update_all_partial_mode;

typedef struct
{
  int init;
  int allowed_cpus;
  int sockets;
  int socket_threads;
} t_phyml_mt_hw_topology;

static const char *PhyML_MT_Backend_Name(t_phyml_mt_backend backend)
{
  switch(backend)
    {
    case PHYML_MT_BACKEND_AVX: return "avx";
    case PHYML_MT_BACKEND_SSE: return "sse";
    default: return "scalar";
    }
}

static const char *PhyML_MT_Phase_Name(t_phyml_mt_phase phase)
{
  switch(phase)
    {
    case PHYML_MT_PHASE_WAVEFRONT: return "update_all_partial_lk";
    case PHYML_MT_PHASE_PARTIAL_LK: return "update_partial_lk";
    case PHYML_MT_PHASE_PMAT: return "update_pmat";
    case PHYML_MT_PHASE_EIGEN: return "update_eigen_lr";
    case PHYML_MT_PHASE_DLK: return "dlk";
    default: return "site_lk";
    }
}

static int PhyML_MT_Read_Allowed_Cpus(unsigned int *cpus, int max_cpus)
{
  FILE *fp;
  char line[4096];

  if(cpus == NULL || max_cpus <= 0) return 0;

  fp = fopen("/proc/self/status","r");
  if(fp == NULL) return 0;

  while(fgets(line,sizeof(line),fp) != NULL)
    {
      if(strncmp(line,"Cpus_allowed_list:",18) == 0)
        {
          const char *p = line + 18;
          int ncpus = 0;

          while(*p != '\0' && *p != '\n')
            {
              char *endptr = NULL;
              long first,last;

              while(*p == ' ' || *p == '\t' || *p == ',') ++p;
              if(*p == '\0' || *p == '\n') break;

              first = strtol(p,&endptr,10);
              if(endptr == p) break;
              last = first;
              p = endptr;

              if(*p == '-')
                {
                  ++p;
                  last = strtol(p,&endptr,10);
                  if(endptr == p) last = first;
                  else p = endptr;
                }

              if(first > last)
                {
                  const long tmp = first;
                  first = last;
                  last = tmp;
                }

              for(long cpu=first; cpu<=last && ncpus<max_cpus; ++cpu)
                cpus[ncpus++] = (unsigned int)cpu;
            }

          fclose(fp);
          return ncpus;
        }
    }

  fclose(fp);
  return 0;
}

static int PhyML_MT_Read_Int_File(const char *path, int *value)
{
  FILE *fp;
  int v;

  if(path == NULL || value == NULL) return NO;

  fp = fopen(path,"r");
  if(fp == NULL) return NO;
  if(fscanf(fp,"%d",&v) != 1)
    {
      fclose(fp);
      return NO;
    }
  fclose(fp);
  *value = v;
  return YES;
}

static const t_phyml_mt_hw_topology *PhyML_MT_Get_HW_Topology(void)
{
  static t_phyml_mt_hw_topology topo = {0,1,1,1};

  if(topo.init == 0)
    {
      unsigned int cpus[1024];
      int ncpus = PhyML_MT_Read_Allowed_Cpus(cpus,(int)(sizeof(cpus)/sizeof(cpus[0])));
      int allowed = 0;
      int sockets[64];
      int nsockets = 0;

      if(ncpus <= 0)
        {
          ncpus = omp_get_num_procs();
          if(ncpus <= 0) ncpus = omp_get_max_threads();
          if(ncpus <= 0) ncpus = 1;
        }

      allowed = ncpus;
      for(int i=0;i<ncpus && i<(int)(sizeof(cpus)/sizeof(cpus[0]));++i)
        {
          char path[256];
          int socket_id = -1;
          int known = NO;

          snprintf(path,sizeof(path),"/sys/devices/system/cpu/cpu%u/topology/physical_package_id",cpus[i]);
          if(PhyML_MT_Read_Int_File(path,&socket_id) == NO) continue;

          for(int j=0;j<nsockets;++j)
            {
              if(sockets[j] == socket_id)
                {
                  known = YES;
                  break;
                }
            }

          if(known == NO && nsockets < (int)(sizeof(sockets)/sizeof(sockets[0])))
            sockets[nsockets++] = socket_id;
        }

      topo.allowed_cpus = MAX(1,allowed);
      topo.sockets = MAX(1,nsockets);
      topo.socket_threads = MAX(1,topo.allowed_cpus / topo.sockets);
      topo.init = 1;
    }

  return &topo;
}

static t_phyml_mt_backend PhyML_MT_Detect_Backend(const t_tree *tree)
{
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
  if(tree != NULL && (tree->mod->ns == 4 || tree->mod->ns == 20)) return PHYML_MT_BACKEND_AVX;
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__ARM_NEON)) && !defined(DISABLE_NATIVE))
  if(tree != NULL && (tree->mod->ns == 4 || tree->mod->ns == 20)) return PHYML_MT_BACKEND_SSE;
#endif
  return PHYML_MT_BACKEND_SCALAR;
}

static int PhyML_MT_Force_Max_Threads(void)
{
  static int init = 0;
  static int force_max = 0;

  if(init == 0)
    {
      const char *env = getenv("PHYML_MT_LK_THREAD_MODE");
      if(env != NULL)
        {
          if(env[0] == 'm' || env[0] == 'M' || env[0] == '1')
            force_max = 1;
        }

      env = getenv("PHYML_MT_LK_FORCE_MAX_THREADS");
      if(env != NULL && env[0] != '\0' && env[0] != '0')
        force_max = 1;

      init = 1;
    }

  return force_max;
}

static int PhyML_MT_Wavefront_Site_Block_Count(int max_width,
                                               int npatterns, int ncatg, int ns,
                                               const t_tree *tree);

static long long PhyML_MT_Base_Min_Work_Per_Thread(t_phyml_mt_backend backend, t_phyml_mt_phase phase)
{
  switch(phase)
    {
    case PHYML_MT_PHASE_WAVEFRONT:
      switch(backend)
        {
        case PHYML_MT_BACKEND_AVX: return 2500000LL;
        case PHYML_MT_BACKEND_SSE: return 1500000LL;
        default: return 1000000LL;
        }

    case PHYML_MT_PHASE_PARTIAL_LK:
      switch(backend)
        {
        case PHYML_MT_BACKEND_AVX: return 320000LL;
        case PHYML_MT_BACKEND_SSE: return 224000LL;
        default: return 160000LL;
        }

    case PHYML_MT_PHASE_PMAT:
      switch(backend)
        {
        case PHYML_MT_BACKEND_AVX: return 50000LL;
        case PHYML_MT_BACKEND_SSE: return 40000LL;
        default: return 25000LL;
        }

    case PHYML_MT_PHASE_EIGEN:
      switch(backend)
        {
        case PHYML_MT_BACKEND_AVX: return 160000LL;
        case PHYML_MT_BACKEND_SSE: return 128000LL;
        default: return 96000LL;
        }

    case PHYML_MT_PHASE_DLK:
      switch(backend)
        {
        case PHYML_MT_BACKEND_AVX: return 800000LL;
        case PHYML_MT_BACKEND_SSE: return 640000LL;
        default: return 480000LL;
        }

    default:
      switch(backend)
        {
        case PHYML_MT_BACKEND_AVX: return 4000LL;
        case PHYML_MT_BACKEND_SSE: return 3000LL;
        default: return 2000LL;
        }
    }
}

static long long PhyML_MT_Min_Work_Per_Thread(t_phyml_mt_backend backend,
                                              t_phyml_mt_phase phase,
                                              int parallelism_cap)
{
  const t_phyml_mt_hw_topology *topo = PhyML_MT_Get_HW_Topology();
  const long long base = PhyML_MT_Base_Min_Work_Per_Thread(backend,phase);
  const int max_threads = MAX(1,omp_get_max_threads());
  long long scaled = base;

  if(topo->allowed_cpus > 8)
    scaled += (base * (long long)(topo->allowed_cpus - 8)) / 24LL;

  if(topo->sockets > 1)
    scaled += (base * (long long)(topo->sockets - 1)) / 4LL;

  if(parallelism_cap > 0 && parallelism_cap < max_threads)
    {
      const int narrow = max_threads - parallelism_cap;
      scaled += (base * (long long)narrow) / (long long)(2 * max_threads);
    }

  return MAX(1LL,scaled);
}

static int PhyML_MT_Socket_Aware_Cap(int threads,
                                     long long work,
                                     int parallelism_cap,
                                     t_phyml_mt_phase phase,
                                     long long min_work_per_thread)
{
  const t_phyml_mt_hw_topology *topo = PhyML_MT_Get_HW_Topology();

  if(topo->sockets <= 1) return threads;
  if(topo->socket_threads <= 0) return threads;
  if(threads <= topo->socket_threads) return threads;

  switch(phase)
    {
    case PHYML_MT_PHASE_PMAT:
      return threads;

    case PHYML_MT_PHASE_WAVEFRONT:
    case PHYML_MT_PHASE_PARTIAL_LK:
    case PHYML_MT_PHASE_SITE_LK:
    case PHYML_MT_PHASE_DLK:
      if((parallelism_cap > 0 && parallelism_cap <= topo->socket_threads * 2) ||
         work < min_work_per_thread * (long long)topo->socket_threads * 2LL)
        return topo->socket_threads;
      break;

    default:
      break;
    }

  return threads;
}

static void PhyML_MT_Debug_Thread_Decision(t_phyml_mt_phase phase,
                                           t_phyml_mt_backend backend,
                                           int max_threads,
                                           int selected_threads,
                                           long long work,
                                           int parallelism_cap,
                                           long long min_work_per_thread)
{
  static int printed[PHYML_MT_PHASE_COUNT][3][2];
  const char *env = getenv("PHYML_MT_LK_DEBUG_THREADS");
  const int force_max = PhyML_MT_Force_Max_Threads();

  if(env == NULL || env[0] == '\0' || env[0] == '0') return;
  if(printed[phase][backend][force_max] == 1) return;
  printed[phase][backend][force_max] = 1;

  PhyML_Fprintf(stderr,
                "\n. MT thread selection: phase=%s backend=%s mode=%s max=%d selected=%d work=%lld cap=%d quantum=%lld\n",
                PhyML_MT_Phase_Name(phase),
                PhyML_MT_Backend_Name(backend),
                force_max ? "max" : "auto",
                max_threads,
                selected_threads,
                work,
                parallelism_cap,
                min_work_per_thread);
}

static int PhyML_MT_Recommended_Threads(long long work,
                                        int parallelism_cap,
                                        t_phyml_mt_backend backend,
                                        t_phyml_mt_phase phase)
{
  const int max_threads = omp_get_max_threads();
  const long long min_work_per_thread = PhyML_MT_Min_Work_Per_Thread(backend,phase,parallelism_cap);
  const int force_max = PhyML_MT_Force_Max_Threads();
  int usable_max_threads = max_threads;
  int threads;

  if(max_threads <= 1)
    {
      PhyML_MT_Debug_Thread_Decision(phase,backend,max_threads,1,work,parallelism_cap,min_work_per_thread);
      return 1;
    }

  if(force_max == 1)
    {
      PhyML_MT_Debug_Thread_Decision(phase,backend,max_threads,max_threads,work,parallelism_cap,min_work_per_thread);
      return max_threads;
    }

  if(parallelism_cap > 0 && usable_max_threads > parallelism_cap) usable_max_threads = parallelism_cap;
  if(usable_max_threads <= 1 || work < min_work_per_thread)
    {
      PhyML_MT_Debug_Thread_Decision(phase,backend,max_threads,1,work,parallelism_cap,min_work_per_thread);
      return 1;
    }

  threads = (int)((work + min_work_per_thread - 1LL) / min_work_per_thread);
  if(threads < 1) threads = 1;
  if(threads > usable_max_threads) threads = usable_max_threads;
  threads = PhyML_MT_Socket_Aware_Cap(threads,work,parallelism_cap,phase,min_work_per_thread);
  PhyML_MT_Debug_Thread_Decision(phase,backend,max_threads,threads,work,parallelism_cap,min_work_per_thread);
  return threads;
}

static int PhyML_MT_Threads_Update_Partial_Lk(int npatterns, int ncatg, int ns)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns;
  return PhyML_MT_Recommended_Threads(work,npatterns,PHYML_MT_BACKEND_SCALAR,PHYML_MT_PHASE_PARTIAL_LK);
}

static t_phyml_mt_update_all_partial_mode PhyML_MT_Select_Update_All_Partial_Lk_Mode(const t_tree *tree)
{
  const t_phyml_mt_backend backend = PhyML_MT_Detect_Backend(tree);

  if(tree == NULL || tree->mod == NULL || tree->mod->s_opt == NULL)
    return PHYML_MT_UPDATE_ALL_PARTIAL_MODE_WAVEFRONT;

  /* Fixed-tree likelihoods benefit from the wavefront cache. The MT8/MT12
     regression shows up during topology search on AVX, where the old
     job-by-job traversal remains better below 16 requested threads. */
  if(tree->mod->s_opt->opt_topo == YES &&
     backend == PHYML_MT_BACKEND_AVX &&
     omp_get_max_threads() < 16)
    return PHYML_MT_UPDATE_ALL_PARTIAL_MODE_LEGACY;

  return PHYML_MT_UPDATE_ALL_PARTIAL_MODE_WAVEFRONT;
}

static int PhyML_MT_Threads_Update_All_Partial_Lk(int njobs, int nlevels, int max_width,
                                                  int npatterns, int ncatg, int ns,
                                                  const t_tree *tree,
                                                  int *site_blocks)
{
  const long long unit_work = (long long)npatterns *
                              (long long)ncatg *
                              (long long)ns *
                              (long long)ns;
  const long long work = (nlevels > 0) ? ((long long)njobs * unit_work + (long long)nlevels - 1LL) / (long long)nlevels : unit_work;
  const t_phyml_mt_backend backend = PhyML_MT_Detect_Backend(tree);
  int blocks = 1;
  int effective_cap = max_width;

  if(site_blocks != NULL) *site_blocks = 1;
  if(max_width > 0)
    {
      blocks = PhyML_MT_Wavefront_Site_Block_Count(max_width,npatterns,ncatg,ns,tree);
      effective_cap = max_width * blocks;
      if(site_blocks != NULL) *site_blocks = blocks;
    }

  return PhyML_MT_Recommended_Threads(work,effective_cap,backend,PHYML_MT_PHASE_WAVEFRONT);
}

static int PhyML_MT_Threads_Site_Lk(int npatterns, int ncatg, int ns, const t_tree *tree)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns;
  const t_phyml_mt_backend backend = PhyML_MT_Detect_Backend(tree);
  return PhyML_MT_Recommended_Threads(work,npatterns,backend,PHYML_MT_PHASE_SITE_LK);
}

static int PhyML_MT_Threads_Update_PMat(int nedges, int ncatg, int ns, const t_tree *tree)
{
  const long long work = (long long)nedges * (long long)ncatg * (long long)ns * (long long)ns * (long long)ns;
  const t_phyml_mt_backend backend = PhyML_MT_Detect_Backend(tree);
  return PhyML_MT_Recommended_Threads(work,nedges,backend,PHYML_MT_PHASE_PMAT);
}

static int PhyML_MT_Threads_dLk(int npatterns, int ncatg, int ns, const t_tree *tree)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns;
  const t_phyml_mt_backend backend = PhyML_MT_Detect_Backend(tree);
  return PhyML_MT_Recommended_Threads(work,npatterns,backend,PHYML_MT_PHASE_DLK);
}

static int PhyML_MT_Threads_Update_Eigen_Lr(int npatterns, int ncatg, int ns, const t_tree *tree)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns;
  const t_phyml_mt_backend backend = PhyML_MT_Detect_Backend(tree);
  return PhyML_MT_Recommended_Threads(work,npatterns,backend,PHYML_MT_PHASE_EIGEN);
}

static int PhyML_MT_Wavefront_Site_Block_Count(int max_width,
                                               int npatterns, int ncatg, int ns,
                                               const t_tree *tree)
{
  const int max_threads = MAX(1,omp_get_max_threads());
  const t_phyml_mt_backend backend = PhyML_MT_Detect_Backend(tree);
  const long long site_work = (long long)ncatg * (long long)ns * (long long)ns;
  const long long min_block_work = MAX(site_work * 16LL,
                                       PhyML_MT_Min_Work_Per_Thread(backend,PHYML_MT_PHASE_PARTIAL_LK,max_width) / 8LL);
  const long long min_sites = MAX(1LL,(min_block_work + site_work - 1LL) / site_work);
  int target_blocks;
  int max_blocks;

  if(max_width <= 0 || npatterns <= 1 || max_threads <= 1) return 1;
  if(max_width >= max_threads) return 1;

  target_blocks = (max_threads + max_width - 1) / max_width;
  if(target_blocks <= 1) return 1;

  max_blocks = (int)((npatterns + min_sites - 1LL) / min_sites);
  if(max_blocks < 1) max_blocks = 1;
  if(target_blocks > max_blocks) target_blocks = max_blocks;
  if(target_blocks > npatterns) target_blocks = npatterns;

  return MAX(1,target_blocks);
}

static double PhyML_MT_Guard_Min_Speedup(void)
{
  const t_phyml_mt_hw_topology *topo = PhyML_MT_Get_HW_Topology();

  if(topo->sockets > 1) return 1.12;
  if(topo->allowed_cpus <= 4) return 1.04;
  return 1.08;
}

static int PhyML_MT_Should_Fallback_ST_Lk(long long pmat_work, int pmat_threads,
                                          long long partial_work, int partial_threads,
                                          long long eigen_work, int eigen_threads,
                                          long long site_work, int site_threads)
{
  double serial_cost = 0.0;
  double mt_cost = 0.0;
  const double min_speedup = PhyML_MT_Guard_Min_Speedup();
  int parallel_phases = 0;
  int max_parallel_threads = 1;

#define PHYML_MT_NOTE_PARALLEL_PHASE(work_, threads_)                 \
  do                                                                  \
    {                                                                 \
      if((work_) > 0 && (threads_) > 1)                               \
        {                                                             \
          ++parallel_phases;                                          \
          if((threads_) > max_parallel_threads)                       \
            max_parallel_threads = (threads_);                        \
        }                                                             \
    }                                                                 \
  while(0)

  if(pmat_work > 0)
    {
      serial_cost += (double)pmat_work;
      mt_cost += (double)pmat_work / (double)MAX(1,pmat_threads);
    }
  if(partial_work > 0)
    {
      serial_cost += (double)partial_work;
      mt_cost += (double)partial_work / (double)MAX(1,partial_threads);
    }
  if(eigen_work > 0)
    {
      serial_cost += (double)eigen_work;
      mt_cost += (double)eigen_work / (double)MAX(1,eigen_threads);
    }
  if(site_work > 0)
    {
      serial_cost += (double)site_work;
      mt_cost += (double)site_work / (double)MAX(1,site_threads);
    }

  PHYML_MT_NOTE_PARALLEL_PHASE(pmat_work,pmat_threads);
  PHYML_MT_NOTE_PARALLEL_PHASE(partial_work,partial_threads);
  PHYML_MT_NOTE_PARALLEL_PHASE(eigen_work,eigen_threads);
  PHYML_MT_NOTE_PARALLEL_PHASE(site_work,site_threads);

  if(serial_cost <= 0.0 || mt_cost <= 0.0) return NO;
  if(parallel_phases <= 1 && max_parallel_threads <= 2)
    return YES;

#undef PHYML_MT_NOTE_PARALLEL_PHASE
  return ((serial_cost / mt_cost) < min_speedup) ? YES : NO;
}

static int PhyML_MT_Should_Fallback_ST_dLk(long long eigen_work, int eigen_threads,
                                           long long dlk_work, int dlk_threads)
{
  return PhyML_MT_Should_Fallback_ST_Lk(0LL,1,0LL,1,eigen_work,eigen_threads,dlk_work,dlk_threads);
}

static int PhyML_MT_Enter_Forced_ST_Mode(int enabled)
{
  int previous_threads = 1;

  if(enabled == NO) return 1;

  previous_threads = omp_get_max_threads();
  if(previous_threads > 1) omp_set_num_threads(1);
  return previous_threads;
}

static void PhyML_MT_Leave_Forced_ST_Mode(int previous_threads)
{
  if(previous_threads > 1) omp_set_num_threads(previous_threads);
}

struct __PhyML_Lk_Update_Job
{
  t_edge *b;
  t_node *d;
  phydbl *dest_plk;
  const phydbl *src_plk1;
  const phydbl *src_plk2;
  int level;
};

static t_lk_thread_ctx *PhyML_MT_Get_Thread_Ctx(t_tree *tree)
{
  int tid = 0;

#if defined(_OPENMP)
  tid = omp_get_thread_num();
#endif

  if(tree->lk_thread_ctx == NULL || tid >= tree->lk_mt_max_threads) return NULL;
  return tree->lk_thread_ctx + tid;
}

static void PhyML_MT_Get_Site_Range(unsigned int nsites, unsigned int *begin, unsigned int *end)
{
  unsigned int tid = 0;
  unsigned int nth = 1;

#if defined(_OPENMP)
  tid = (unsigned int)omp_get_thread_num();
  nth = (unsigned int)omp_get_num_threads();
#endif

  *begin = (unsigned int)(((unsigned long long)nsites * tid) / nth);
  *end   = (unsigned int)(((unsigned long long)nsites * (tid + 1U)) / nth);
}

static void PhyML_Debug_Update_Partial_Lk_Wavefront(const t_lk_update_job *jobs, int njobs, const int *level_offsets, int nlevels)
{
  const char *env = getenv("PHYML_MT_LK_WAVEFRONT_DEBUG");

  (void)jobs;

  if(env == NULL || env[0] == '\0' || env[0] == '0') return;

  if(njobs > 0 && level_offsets != NULL && nlevels > 0)
    {
      int level;
      int max_width = 0;
      long long total_width = 0;

      for(level=0;level<nlevels;++level)
        {
          const int width = level_offsets[level+1] - level_offsets[level];
          if(width > max_width) max_width = width;
          total_width += width;
        }

      PhyML_Fprintf(stderr,
                    "\n. Update_Partial_Lk wavefront stats: jobs=%d levels=%d max_width=%d mean_width=%.2f barriers_before=%d barriers_after=%d\n",
                    njobs,
                    nlevels,
                    max_width,
                    (nlevels > 0) ? ((phydbl)total_width / (phydbl)nlevels) : .0,
                    njobs,
                    nlevels);
    }
}

static int PhyML_Update_All_Partial_Lk_Max_Width(const int *level_offsets, int nlevels)
{
  int level;
  int max_width = 0;

  if(level_offsets == NULL || nlevels <= 0) return 0;

  for(level=0;level<nlevels;++level)
    {
      const int width = level_offsets[level+1] - level_offsets[level];
      if(width > max_width) max_width = width;
    }

  return max_width;
}

static int PhyML_MT_Collect_Warnings(t_tree *tree, int nthreads);
static void PhyML_Update_Eigen_Lr_Team(t_edge *b, t_tree *tree, t_lk_thread_ctx *ctx);
static void PhyML_Update_Eigen_And_Lk_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx);
static void PhyML_Lk_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx);
#if PHYML_MT_LK_RUNTIME && PHYML_OPT_PARTIAL_LK
static void PhyML_Update_Partial_Lk_Team(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx);
#endif
static int PhyML_Can_Use_Update_Partial_Lk_Wavefront(const t_tree *tree);
static void PhyML_Fill_Update_Partial_Lk_Job_Metadata(t_tree *tree, t_lk_update_job *job);
static int PhyML_Build_Update_All_Partial_Lk_Wavefront(t_tree *tree, t_lk_update_job *jobs, int njobs, int *level_offsets);
static void PhyML_Execute_Update_Partial_Lk_Job(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx);
static void PhyML_Execute_Update_Partial_Lk_Job_Range(t_tree *tree, t_edge *b, t_node *d,
                                                      unsigned int site_begin, unsigned int site_end,
                                                      t_lk_thread_ctx *ctx);
static void PhyML_Update_All_Partial_Lk_Legacy_Team(t_tree *tree, const t_lk_update_job *jobs, int njobs, t_lk_thread_ctx *ctx);
static void PhyML_Update_All_Partial_Lk_Legacy_MT(t_tree *tree, const t_lk_update_job *jobs, int njobs, int nthreads);
static void PhyML_Update_All_Partial_Lk_Wavefront_Team(t_tree *tree, const t_lk_update_job *jobs,
                                                       const int *level_offsets, int nlevels,
                                                       int active_threads, int site_blocks,
                                                       t_lk_thread_ctx *ctx);
static void PhyML_Update_All_Partial_Lk_Wavefront_MT(t_tree *tree, const t_lk_update_job *jobs,
                                                     const int *level_offsets, int nlevels,
                                                     int nthreads, int site_blocks);
static unsigned long long PhyML_Update_All_Partial_Lk_Wavefront_Signature(const t_tree *tree);
static void PhyML_Clear_Update_All_Partial_Lk_Wavefront_Cache(t_tree *tree);
static int PhyML_Ensure_Update_All_Partial_Lk_Wavefront_Cache(t_tree *tree);
#if PHYML_MT_LK_RUNTIME && PHYML_OPT_PARTIAL_LK
static void Default_Update_Partial_Lk_Team(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx);
static void Default_Update_Partial_Lk_Range(t_tree *tree, t_edge *b, t_node *d,
                                            unsigned int site_begin, unsigned int site_end);
static void Core_Default_Update_Partial_Lk_Team(const t_node *n_v1, const t_node *n_v2,
                                                phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                const phydbl *Pij1, const phydbl *Pij2,
                                                int *sum_scale0, const int *sum_scale1, const int *sum_scale2,
                                                const int ns, const int ncatg, const int npatterns, const int apply_scaling,
                                                const phydbl *wght);
static void Core_Default_Update_Partial_Lk_Range(const t_node *n_v1, const t_node *n_v2,
                                                 phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                 const phydbl *Pij1, const phydbl *Pij2,
                                                 int *sum_scale0, const int *sum_scale1, const int *sum_scale2,
                                                 const int ns, const int ncatg,
                                                 unsigned int site_begin, unsigned int site_end,
                                                 const int apply_scaling, const phydbl *wght);
#endif

static t_edge *PhyML_Lk_Target_Edge(t_tree *tree)
{
  if(tree->n_root)
    {
      if(tree->ignore_root == NO)
        return (tree->n_root->v[1]->tax == NO) ? (tree->n_root->b[2]) : (tree->n_root->b[1]);
      else
        return tree->e_root;
    }

  return tree->a_nodes[tree->tip_root]->b[0];
}

static void PhyML_Append_Post_Order_Jobs(t_node *a, t_node *d, t_tree *tree, t_lk_update_job *jobs, int *njobs)
{
  int i,dir;

  dir = -1;
  if(d->tax) return;

  if(tree->n_root != NULL)
    {
      for(i=0;i<3;++i)
        {
          if(d->v[i] != a && !(a == tree->n_root && d->b[i] == tree->e_root))
            PhyML_Append_Post_Order_Jobs(d,d->v[i],tree,jobs,njobs);
          else
            dir = i;
        }
    }
  else
    {
      for(i=0;i<3;++i)
        {
          if(d->v[i] != a) PhyML_Append_Post_Order_Jobs(d,d->v[i],tree,jobs,njobs);
          else dir = i;
        }
    }

  if(tree->ignore_root == NO && d->b[dir] == tree->e_root)
    {
      jobs[*njobs].b = (d == tree->n_root->v[1]) ? tree->n_root->b[1] : tree->n_root->b[2];
      jobs[*njobs].d = d;
      ++(*njobs);
    }
  else
    {
      jobs[*njobs].b = d->b[dir];
      jobs[*njobs].d = d;
      ++(*njobs);
    }
}

static void PhyML_Append_Pre_Order_Jobs(t_node *a, t_node *d, t_tree *tree, t_lk_update_job *jobs, int *njobs)
{
  int i;

  if(d->tax) return;

  if(tree->n_root)
    {
      for(i=0;i<3;++i)
        {
          if(d->v[i] != a && !(a == tree->n_root && d->b[i] == tree->e_root))
            {
              jobs[*njobs].b = d->b[i];
              jobs[*njobs].d = d;
              ++(*njobs);
              PhyML_Append_Pre_Order_Jobs(d,d->v[i],tree,jobs,njobs);
            }
        }
    }
  else
    {
      for(i=0;i<3;++i)
        {
          if(d->v[i] != a)
            {
              jobs[*njobs].b = d->b[i];
              jobs[*njobs].d = d;
              ++(*njobs);
              PhyML_Append_Pre_Order_Jobs(d,d->v[i],tree,jobs,njobs);
            }
        }
    }
}

static int PhyML_Build_Update_All_Partial_Lk_Jobs(t_tree *tree, t_lk_update_job *jobs)
{
  int njobs = 0;

  if(tree->n_root)
    {
      if(tree->ignore_root == NO)
        {
          PhyML_Append_Post_Order_Jobs(tree->n_root,tree->n_root->v[1],tree,jobs,&njobs);
          PhyML_Append_Post_Order_Jobs(tree->n_root,tree->n_root->v[2],tree,jobs,&njobs);

          jobs[njobs].b = tree->n_root->b[1];
          jobs[njobs].d = tree->n_root;
          ++njobs;
          jobs[njobs].b = tree->n_root->b[2];
          jobs[njobs].d = tree->n_root;
          ++njobs;

          if(tree->both_sides == YES)
            {
              PhyML_Append_Pre_Order_Jobs(tree->n_root,tree->n_root->v[2],tree,jobs,&njobs);
              PhyML_Append_Pre_Order_Jobs(tree->n_root,tree->n_root->v[1],tree,jobs,&njobs);
            }
        }
      else
        {
          PhyML_Append_Post_Order_Jobs(tree->e_root->rght,tree->e_root->left,tree,jobs,&njobs);
          PhyML_Append_Post_Order_Jobs(tree->e_root->left,tree->e_root->rght,tree,jobs,&njobs);

          if(tree->both_sides == YES)
            {
              PhyML_Append_Pre_Order_Jobs(tree->e_root->rght,tree->e_root->left,tree,jobs,&njobs);
              PhyML_Append_Pre_Order_Jobs(tree->e_root->left,tree->e_root->rght,tree,jobs,&njobs);
            }
        }
    }
  else
    {
      PhyML_Append_Post_Order_Jobs(tree->a_nodes[tree->tip_root],tree->a_nodes[tree->tip_root]->v[0],tree,jobs,&njobs);
      if(tree->both_sides == YES)
        PhyML_Append_Pre_Order_Jobs(tree->a_nodes[tree->tip_root],tree->a_nodes[tree->tip_root]->v[0],tree,jobs,&njobs);
    }

  return njobs;
}

static int PhyML_Can_Use_Update_Partial_Lk_Wavefront(const t_tree *tree)
{
#if !PHYML_OPT_PARTIAL_LK
  (void)tree;
  return NO;
#else
  if(tree == NULL) return NO;
  if(tree->is_mixt_tree == YES) return NO;
  if(tree->mod->use_m4mod != NO) return NO;
  if(tree->mod->ns != 4 && tree->mod->ns != 20) return NO;
  if(tree->update_alias_subpatt == YES && tree->io->do_alias_subpatt == YES) return NO;
  return YES;
#endif
}

static void PhyML_Fill_Update_Partial_Lk_Job_Metadata(t_tree *tree, t_lk_update_job *job)
{
  t_node *n_v1, *n_v2;
  phydbl *p_lk,*p_lk_v1,*p_lk_v2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;

  n_v1 = n_v2                 = NULL;
  p_lk = p_lk_v1 = p_lk_v2    = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  sum_scale = sum_scale_v1 = sum_scale_v2 = NULL;
  p_lk_loc                    = NULL;

  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &p_lk,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&p_lk_v1,&sum_scale_v1,
                     &Pij2,&tPij2,&p_lk_v2,&sum_scale_v2,
                     job->d,job->b,tree);

  job->dest_plk = p_lk;
  job->src_plk1 = p_lk_v1;
  job->src_plk2 = p_lk_v2;
  job->level = 0;
}

static int PhyML_Find_Update_Partial_Lk_Producer(const t_lk_update_job *jobs, int njobs, const phydbl *ptr)
{
  int i;

  if(ptr == NULL) return -1;

  for(i=0;i<njobs;++i)
    if(jobs[i].dest_plk == ptr)
      return i;

  return -1;
}

static inline unsigned long long PhyML_Wavefront_Hash_Combine(unsigned long long hash, uintptr_t value)
{
  hash ^= (unsigned long long)value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

static unsigned long long PhyML_Update_All_Partial_Lk_Wavefront_Signature(const t_tree *tree)
{
  unsigned long long signature = 1469598103934665603ULL;
  int i;

  if(tree == NULL) return 0ULL;

  signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)tree->n_root);
  signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)tree->e_root);
  signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)tree->tip_root);
  signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)tree->ignore_root);
  signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)tree->both_sides);

  for(i=0;i<2*tree->n_otu-1;++i)
    {
      const t_edge *edge = tree->a_edges[i];

      if(edge == NULL) continue;

      signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)edge->num);
      signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)(edge->left ? edge->left->num + 1 : 0));
      signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)(edge->rght ? edge->rght->num + 1 : 0));
      signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)edge->p_lk_left);
      signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)edge->p_lk_rght);
      signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)edge->p_lk_tip_l);
      signature = PhyML_Wavefront_Hash_Combine(signature,(uintptr_t)edge->p_lk_tip_r);
    }

  return (signature == 0ULL) ? 1ULL : signature;
}

static void PhyML_Clear_Update_All_Partial_Lk_Wavefront_Cache(t_tree *tree)
{
  if(tree == NULL) return;

  Free(tree->lk_wavefront_level_offsets);
  Free(tree->lk_wavefront_jobs);
  tree->lk_wavefront_level_offsets = NULL;
  tree->lk_wavefront_jobs = NULL;
  tree->lk_wavefront_njobs = 0;
  tree->lk_wavefront_nlevels = 0;
  tree->lk_wavefront_max_width = 0;
  tree->lk_wavefront_valid = NO;
  tree->lk_wavefront_signature = 0ULL;
}

static int PhyML_Ensure_Update_All_Partial_Lk_Wavefront_Cache(t_tree *tree)
{
  unsigned long long signature;
  int max_jobs;

  if(tree == NULL) return NO;
  signature = PhyML_Update_All_Partial_Lk_Wavefront_Signature(tree);
  max_jobs = 8 * tree->n_otu + 8;

  if(tree->lk_wavefront_valid == YES &&
     tree->lk_wavefront_signature == signature &&
     tree->lk_wavefront_jobs != NULL &&
     tree->lk_wavefront_level_offsets != NULL)
    return YES;

  PhyML_Clear_Update_All_Partial_Lk_Wavefront_Cache(tree);

  tree->lk_wavefront_jobs = (t_lk_update_job *)mCalloc((size_t)max_jobs,sizeof(t_lk_update_job));
  tree->lk_wavefront_njobs = PhyML_Build_Update_All_Partial_Lk_Jobs(tree,tree->lk_wavefront_jobs);
  assert(tree->lk_wavefront_njobs <= max_jobs);
  if(tree->lk_wavefront_njobs <= 0)
    {
      PhyML_Clear_Update_All_Partial_Lk_Wavefront_Cache(tree);
      return NO;
    }

  tree->lk_wavefront_level_offsets = (int *)mCalloc((size_t)(tree->lk_wavefront_njobs + 1),sizeof(int));
  tree->lk_wavefront_nlevels = PhyML_Build_Update_All_Partial_Lk_Wavefront(tree,
                                                                            tree->lk_wavefront_jobs,
                                                                            tree->lk_wavefront_njobs,
                                                                            tree->lk_wavefront_level_offsets);
  tree->lk_wavefront_max_width = PhyML_Update_All_Partial_Lk_Max_Width(tree->lk_wavefront_level_offsets,
                                                                       tree->lk_wavefront_nlevels);
  tree->lk_wavefront_signature = signature;
  tree->lk_wavefront_valid = YES;

  PhyML_Debug_Update_Partial_Lk_Wavefront(tree->lk_wavefront_jobs,
                                          tree->lk_wavefront_njobs,
                                          tree->lk_wavefront_level_offsets,
                                          tree->lk_wavefront_nlevels);
  return YES;
}

static int PhyML_Build_Update_All_Partial_Lk_Wavefront(t_tree *tree, t_lk_update_job *jobs, int njobs, int *level_offsets)
{
  int *indegree;
  int *succ_counts;
  int *succ_offsets;
  int *succ_next;
  int *succ_list;
  int *ready;
  t_lk_update_job *sorted_jobs;
  int total_edges = 0;
  int i;
  int nlevels = 0;
  int out_idx = 0;
  int nprocessed = 0;

  indegree = (int *)mCalloc((size_t)njobs,sizeof(int));
  succ_counts = (int *)mCalloc((size_t)njobs,sizeof(int));
  succ_offsets = (int *)mCalloc((size_t)(njobs + 1),sizeof(int));
  succ_next = (int *)mCalloc((size_t)njobs,sizeof(int));
  ready = (int *)mCalloc((size_t)njobs,sizeof(int));
  sorted_jobs = (t_lk_update_job *)mCalloc((size_t)njobs,sizeof(t_lk_update_job));

  for(i=0;i<njobs;++i)
    {
      int prod1,prod2;

      PhyML_Fill_Update_Partial_Lk_Job_Metadata(tree,jobs + i);
      prod1 = PhyML_Find_Update_Partial_Lk_Producer(jobs,njobs,jobs[i].src_plk1);
      prod2 = PhyML_Find_Update_Partial_Lk_Producer(jobs,njobs,jobs[i].src_plk2);

      if(prod1 >= 0)
        {
          ++indegree[i];
          ++succ_counts[prod1];
          ++total_edges;
        }
      if(prod2 >= 0 && prod2 != prod1)
        {
          ++indegree[i];
          ++succ_counts[prod2];
          ++total_edges;
        }
    }

  for(i=0;i<njobs;++i) succ_offsets[i+1] = succ_offsets[i] + succ_counts[i];
  succ_list = (int *)mCalloc((size_t)MAX(total_edges,1),sizeof(int));
  for(i=0;i<njobs;++i) succ_next[i] = succ_offsets[i];

  for(i=0;i<njobs;++i)
    {
      int prod1 = PhyML_Find_Update_Partial_Lk_Producer(jobs,njobs,jobs[i].src_plk1);
      int prod2 = PhyML_Find_Update_Partial_Lk_Producer(jobs,njobs,jobs[i].src_plk2);

      if(prod1 >= 0) succ_list[succ_next[prod1]++] = i;
      if(prod2 >= 0 && prod2 != prod1) succ_list[succ_next[prod2]++] = i;
    }

  while(nprocessed < njobs)
    {
      int ready_count = 0;

      level_offsets[nlevels] = out_idx;
      for(i=0;i<njobs;++i)
        {
          if(indegree[i] == 0)
            {
              ready[ready_count++] = i;
              indegree[i] = -1;
            }
        }

      if(ready_count == 0)
        {
          PhyML_Fprintf(stderr,"\n. Cyclic dependency detected while building partial-likelihood wavefront.\n");
          assert(FALSE);
        }

      for(i=0;i<ready_count;++i)
        {
          const int job_idx = ready[i];
          int succ_it;

          jobs[job_idx].level = nlevels;
          sorted_jobs[out_idx] = jobs[job_idx];
          ++out_idx;
          ++nprocessed;

          for(succ_it=succ_offsets[job_idx]; succ_it<succ_offsets[job_idx+1]; ++succ_it)
            {
              const int succ_idx = succ_list[succ_it];
              assert(indegree[succ_idx] > 0);
              --indegree[succ_idx];
            }
        }

      ++nlevels;
    }

  level_offsets[nlevels] = out_idx;
  for(i=0;i<njobs;++i) jobs[i] = sorted_jobs[i];

  Free(sorted_jobs);
  Free(ready);
  Free(succ_list);
  Free(succ_next);
  Free(succ_offsets);
  Free(succ_counts);
  Free(indegree);

  return nlevels;
}

static void PhyML_Execute_Update_Partial_Lk_Job(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx)
{
  assert(ctx != NULL);

  if(b->left == d && b->update_partial_lk_left == NO) return;
  if(b->rght == d && b->update_partial_lk_rght == NO) return;
  if(d->tax) return;

#if PHYML_OPT_PARTIAL_LK && ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
  AVX_Update_Partial_Lk_Wavefront_Job(tree,b,d,ctx);
#elif PHYML_OPT_PARTIAL_LK && ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
  SSE_Update_Partial_Lk_Wavefront_Job(tree,b,d,ctx);
#else
  Default_Update_Partial_Lk(tree,b,d);
#endif
}

static void PhyML_Execute_Update_Partial_Lk_Job_Range(t_tree *tree, t_edge *b, t_node *d,
                                                      unsigned int site_begin, unsigned int site_end,
                                                      t_lk_thread_ctx *ctx)
{
  assert(ctx != NULL);
  if(site_begin >= site_end) return;

  if(b->left == d && b->update_partial_lk_left == NO) return;
  if(b->rght == d && b->update_partial_lk_rght == NO) return;
  if(d->tax) return;

#if PHYML_OPT_PARTIAL_LK && ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
  AVX_Update_Partial_Lk_Wavefront_Job_Range(tree,b,d,site_begin,site_end,ctx);
#elif PHYML_OPT_PARTIAL_LK && ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
  SSE_Update_Partial_Lk_Wavefront_Job_Range(tree,b,d,site_begin,site_end,ctx);
#else
  Default_Update_Partial_Lk_Range(tree,b,d,site_begin,site_end);
#endif
}

static void PhyML_Update_All_Partial_Lk_Legacy_Team(t_tree *tree, const t_lk_update_job *jobs, int njobs, t_lk_thread_ctx *ctx)
{
  int job;

  assert(ctx != NULL);

  for(job=0;job<njobs;++job)
    {
      PhyML_Update_Partial_Lk_Team(tree,jobs[job].b,jobs[job].d,ctx);
    }
}

static void PhyML_Update_All_Partial_Lk_Legacy_MT(t_tree *tree, const t_lk_update_job *jobs, int njobs, int nthreads)
{
  #pragma omp parallel num_threads(nthreads)
    {
      t_lk_thread_ctx *ctx = PhyML_MT_Get_Thread_Ctx(tree);

      assert(ctx != NULL);
      PhyML_Update_All_Partial_Lk_Legacy_Team(tree,jobs,njobs,ctx);
    }
}

static void PhyML_Update_All_Partial_Lk_Wavefront_Team(t_tree *tree, const t_lk_update_job *jobs,
                                                       const int *level_offsets, int nlevels,
                                                       int active_threads, int site_blocks,
                                                       t_lk_thread_ctx *ctx)
{
  int tid = 0;
  int team_threads = 1;
  int level;

  assert(ctx != NULL);
#if defined(_OPENMP)
  tid = omp_get_thread_num();
  team_threads = omp_get_num_threads();
#endif
  if(active_threads < 1) active_threads = 1;
  if(active_threads > team_threads) active_threads = team_threads;
  if(site_blocks < 1) site_blocks = 1;

  for(level=0;level<nlevels;++level)
    {
      const int begin = level_offsets[level];
      const int end = level_offsets[level+1];
      const int width = end - begin;

      if(tid < active_threads)
        {
          if(site_blocks <= 1)
            {
              const int local_begin = (width * tid) / active_threads;
              const int local_end = (width * (tid + 1)) / active_threads;

              for(int job=begin + local_begin; job<begin + local_end; ++job)
                PhyML_Execute_Update_Partial_Lk_Job(tree,jobs[job].b,jobs[job].d,ctx);
            }
          else
            {
              const int total_units = width * site_blocks;
              const int local_begin = (total_units * tid) / active_threads;
              const int local_end = (total_units * (tid + 1)) / active_threads;

              for(int unit=local_begin; unit<local_end; ++unit)
                {
                  const int block_idx = unit / width;
                  const int job_idx = begin + (unit % width);
                  const unsigned int site_begin = (unsigned int)(((unsigned long long)tree->n_pattern * (unsigned int)block_idx) / (unsigned int)site_blocks);
                  const unsigned int site_end = (unsigned int)(((unsigned long long)tree->n_pattern * (unsigned int)(block_idx + 1)) / (unsigned int)site_blocks);

                  PhyML_Execute_Update_Partial_Lk_Job_Range(tree,jobs[job_idx].b,jobs[job_idx].d,site_begin,site_end,ctx);
                }
            }
        }

      #pragma omp barrier
    }
}

static void PhyML_Update_All_Partial_Lk_Wavefront_MT(t_tree *tree, const t_lk_update_job *jobs,
                                                     const int *level_offsets, int nlevels,
                                                     int nthreads, int site_blocks)
{
  #pragma omp parallel num_threads(nthreads)
    {
      t_lk_thread_ctx *ctx = PhyML_MT_Get_Thread_Ctx(tree);

      assert(ctx != NULL);
      PhyML_Update_All_Partial_Lk_Wavefront_Team(tree,jobs,level_offsets,nlevels,nthreads,site_blocks,ctx);
    }
}
#endif

#if PHYML_OPT_PARTIAL_LK
static inline void Partial_Lk_Inin_4(const phydbl *Pij1, const phydbl *plk1,
                                     const phydbl *Pij2, const phydbl *plk2,
                                     phydbl *plk0);
static inline void Partial_Lk_Exex_4(const phydbl *Pij1, const int state1,
                                     const phydbl *Pij2, const int state2,
                                     phydbl *plk0);
static inline void Partial_Lk_Exin_4(const phydbl *Pij1, const int state1,
                                     const phydbl *Pij2, const phydbl *plk2,
                                     phydbl *plk0);
static inline void Partial_Lk_Inin_20(const phydbl *Pij1, const phydbl *plk1,
                                      const phydbl *Pij2, const phydbl *plk2,
                                      phydbl *plk0);
static inline void Partial_Lk_Exex_20(const phydbl *Pij1, const int state1,
                                      const phydbl *Pij2, const int state2,
                                      phydbl *plk0);
static inline void Partial_Lk_Exin_20(const phydbl *Pij1, const int state1,
                                      const phydbl *Pij2, const phydbl *plk2,
                                      phydbl *plk0);
static inline phydbl Partial_Lk_Inin_4_Max(const phydbl *Pij1, const phydbl *plk1,
                                           const phydbl *Pij2, const phydbl *plk2,
                                           phydbl *plk0);
static inline phydbl Partial_Lk_Exex_4_Max(const phydbl *Pij1, const int state1,
                                           const phydbl *Pij2, const int state2,
                                           phydbl *plk0);
static inline phydbl Partial_Lk_Exin_4_Max(const phydbl *Pij1, const int state1,
                                           const phydbl *Pij2, const phydbl *plk2,
                                           phydbl *plk0);
static inline phydbl Partial_Lk_Inin_20_Max(const phydbl *Pij1, const phydbl *plk1,
                                            const phydbl *Pij2, const phydbl *plk2,
                                            phydbl *plk0);
static inline phydbl Partial_Lk_Exex_20_Max(const phydbl *Pij1, const int state1,
                                            const phydbl *Pij2, const int state2,
                                            phydbl *plk0);
static inline phydbl Partial_Lk_Exin_20_Max(const phydbl *Pij1, const int state1,
                                            const phydbl *Pij2, const phydbl *plk2,
                                            phydbl *plk0);
#endif




//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Tips_At_One_Site_Nucleotides_Float(char state, int pos, phydbl *p_lk)
{
  switch(state)
    {
    case 'A' : p_lk[pos+0]=1.; p_lk[pos+1]=p_lk[pos+2]=p_lk[pos+3]=.0;
      break;
    case 'C' : p_lk[pos+1]=1.; p_lk[pos+0]=p_lk[pos+2]=p_lk[pos+3]=.0;
      break;
    case 'G' : p_lk[pos+2]=1.; p_lk[pos+1]=p_lk[pos+0]=p_lk[pos+3]=.0;
      break;
    case 'T' : p_lk[pos+3]=1.; p_lk[pos+1]=p_lk[pos+2]=p_lk[pos+0]=.0;
      break;
    case 'U' : p_lk[pos+3]=1.; p_lk[pos+1]=p_lk[pos+2]=p_lk[pos+0]=.0;
      break;
    case 'M' : p_lk[pos+0]=p_lk[pos+1]=1.; p_lk[pos+2]=p_lk[pos+3]=.0;
      break;
    case 'R' : p_lk[pos+0]=p_lk[pos+2]=1.; p_lk[pos+1]=p_lk[pos+3]=.0;
      break;
    case 'W' : p_lk[pos+0]=p_lk[pos+3]=1.; p_lk[pos+1]=p_lk[pos+2]=.0;
      break;
    case 'S' : p_lk[pos+1]=p_lk[pos+2]=1.; p_lk[pos+0]=p_lk[pos+3]=.0;
      break;
    case 'Y' : p_lk[pos+1]=p_lk[pos+3]=1.; p_lk[pos+0]=p_lk[pos+2]=.0;
      break;
    case 'K' : p_lk[pos+2]=p_lk[pos+3]=1.; p_lk[pos+0]=p_lk[pos+1]=.0;
      break;
    case 'B' : p_lk[pos+1]=p_lk[pos+2]=p_lk[pos+3]=1.; p_lk[pos+0]=.0;
      break;
    case 'D' : p_lk[pos+0]=p_lk[pos+2]=p_lk[pos+3]=1.; p_lk[pos+1]=.0;
      break;
    case 'H' : p_lk[pos+0]=p_lk[pos+1]=p_lk[pos+3]=1.; p_lk[pos+2]=.0;
      break;
    case 'V' : p_lk[pos+0]=p_lk[pos+1]=p_lk[pos+2]=1.; p_lk[pos+3]=.0;
      break;
    case 'N' : case 'X' : case '?' : case 'O' : case '-' :
      p_lk[pos+0]=p_lk[pos+1]=p_lk[pos+2]=p_lk[pos+3]=1.;break;
    default :
      {
        PhyML_Fprintf(stderr,"\n. Unknown character state : '%c'.\n",state);
        Exit("\n. Init failed (data type supposed to be DNA)\n");
        break;
      }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Tips_At_One_Site_Nucleotides_Int(char state, int pos, short int *p_pars)
{
  switch(state)
    {
    case 'A' : p_pars[pos+0]=1; p_pars[pos+1]=p_pars[pos+2]=p_pars[pos+3]=0;
      break;
    case 'C' : p_pars[pos+1]=1; p_pars[pos+0]=p_pars[pos+2]=p_pars[pos+3]=0;
      break;
    case 'G' : p_pars[pos+2]=1; p_pars[pos+1]=p_pars[pos+0]=p_pars[pos+3]=0;
      break;
    case 'T' : p_pars[pos+3]=1; p_pars[pos+1]=p_pars[pos+2]=p_pars[pos+0]=0;
      break;
    case 'U' : p_pars[pos+3]=1; p_pars[pos+1]=p_pars[pos+2]=p_pars[pos+0]=0;
      break;
    case 'M' : p_pars[pos+0]=p_pars[pos+1]=1; p_pars[pos+2]=p_pars[pos+3]=0;
      break;
    case 'R' : p_pars[pos+0]=p_pars[pos+2]=1; p_pars[pos+1]=p_pars[pos+3]=0;
      break;
    case 'W' : p_pars[pos+0]=p_pars[pos+3]=1; p_pars[pos+1]=p_pars[pos+2]=0;
      break;
    case 'S' : p_pars[pos+1]=p_pars[pos+2]=1; p_pars[pos+0]=p_pars[pos+3]=0;
      break;
    case 'Y' : p_pars[pos+1]=p_pars[pos+3]=1; p_pars[pos+0]=p_pars[pos+2]=0;
      break;
    case 'K' : p_pars[pos+2]=p_pars[pos+3]=1; p_pars[pos+0]=p_pars[pos+1]=0;
      break;
    case 'B' : p_pars[pos+1]=p_pars[pos+2]=p_pars[pos+3]=1; p_pars[pos+0]=0;
      break;
    case 'D' : p_pars[pos+0]=p_pars[pos+2]=p_pars[pos+3]=1; p_pars[pos+1]=0;
      break;
    case 'H' : p_pars[pos+0]=p_pars[pos+1]=p_pars[pos+3]=1; p_pars[pos+2]=0;
      break;
    case 'V' : p_pars[pos+0]=p_pars[pos+1]=p_pars[pos+2]=1; p_pars[pos+3]=0;
      break;
    case 'N' : case 'X' : case '?' : case 'O' : case '-' :
      p_pars[pos+0]=p_pars[pos+1]=p_pars[pos+2]=p_pars[pos+3]=1;break;
    default :
      {
        PhyML_Fprintf(stderr,"\n. Unknown character state : '%c'.\n",state);
        Exit("\n. Init failed (data type supposed to be DNA)\n");
        break;
      }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Tips_At_One_Site_AA_Float(char aa, int pos, phydbl *p_lk)
{
  int i;

  for(i=0;i<20;i++) p_lk[pos+i] = .0;

  switch(aa){
  case 'A' : p_lk[pos+0]= 1.; break;/* Alanine */
  case 'R' : p_lk[pos+1]= 1.; break;/* Arginine */
  case 'N' : p_lk[pos+2]= 1.; break;/* Asparagine */
  case 'D' : p_lk[pos+3]= 1.; break;/* Aspartic acid */
  case 'C' : p_lk[pos+4]= 1.; break;/* Cysteine */
  case 'Q' : p_lk[pos+5]= 1.; break;/* Glutamine */
  case 'E' : p_lk[pos+6]= 1.; break;/* Glutamic acid */
  case 'G' : p_lk[pos+7]= 1.; break;/* Glycine */
  case 'H' : p_lk[pos+8]= 1.; break;/* Histidine */
  case 'I' : p_lk[pos+9]= 1.; break;/* Isoleucine */
  case 'L' : p_lk[pos+10]=1.; break;/* Leucine */
  case 'K' : p_lk[pos+11]=1.; break;/* Lysine */
  case 'M' : p_lk[pos+12]=1.; break;/* Methionine */
  case 'F' : p_lk[pos+13]=1.; break;/* Phenylalanin */
  case 'P' : p_lk[pos+14]=1.; break;/* Proline */
  case 'S' : p_lk[pos+15]=1.; break;/* Serine */
  case 'T' : p_lk[pos+16]=1.; break;/* Threonine */
  case 'W' : p_lk[pos+17]=1.; break;/* Tryptophan */
  case 'Y' : p_lk[pos+18]=1.; break;/* Tyrosine */
  case 'V' : p_lk[pos+19]=1.; break;/* Valine */

  case 'B' : p_lk[pos+2]= 1.; break;/* Asparagine */
  case 'Z' : p_lk[pos+5]= 1.; break;/* Glutamine */

  case 'X' : case '?' : case '-' : for(i=0;i<20;i++) p_lk[pos+i] = 1.; break;
  default :
    {
      PhyML_Fprintf(stderr,"\n. Unknown character state : '%c'.\n",aa);
      Exit("\n. Init failed (data type supposed to be amino-acids)\n");
      break;
    }
  }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Tips_At_One_Site_AA_Int(char aa, int pos, short int *p_pars)
{
  int i;

  for(i=0;i<20;i++) p_pars[pos+i] = .0;

  switch(aa){
  case 'A' : p_pars[pos+0]  = 1; break;/* Alanine */
  case 'R' : p_pars[pos+1]  = 1; break;/* Arginine */
  case 'N' : p_pars[pos+2]  = 1; break;/* Asparagine */
  case 'D' : p_pars[pos+3]  = 1; break;/* Aspartic acid */
  case 'C' : p_pars[pos+4]  = 1; break;/* Cysteine */
  case 'Q' : p_pars[pos+5]  = 1; break;/* Glutamine */
  case 'E' : p_pars[pos+6]  = 1; break;/* Glutamic acid */
  case 'G' : p_pars[pos+7]  = 1; break;/* Glycine */
  case 'H' : p_pars[pos+8]  = 1; break;/* Histidine */
  case 'I' : p_pars[pos+9]  = 1; break;/* Isoleucine */
  case 'L' : p_pars[pos+10] = 1; break;/* Leucine */
  case 'K' : p_pars[pos+11] = 1; break;/* Lysine */
  case 'M' : p_pars[pos+12] = 1; break;/* Methionine */
  case 'F' : p_pars[pos+13] = 1; break;/* Phenylalanin */
  case 'P' : p_pars[pos+14] = 1; break;/* Proline */
  case 'S' : p_pars[pos+15] = 1; break;/* Serine */
  case 'T' : p_pars[pos+16] = 1; break;/* Threonine */
  case 'W' : p_pars[pos+17] = 1; break;/* Tryptophan */
  case 'Y' : p_pars[pos+18] = 1; break;/* Tyrosine */
  case 'V' : p_pars[pos+19] = 1; break;/* Valine */

  case 'B' : p_pars[pos+2]  = 1; break;/* Asparagine */
  case 'Z' : p_pars[pos+5]  = 1; break;/* Glutamine */

  case 'X' : case '?' : case '-' : for(i=0;i<20;i++) p_pars[pos+i] = 1; break;
  default :
    {
      PhyML_Fprintf(stderr,"\n. Unknown character state : '%c'.\n",aa);
      Exit("\n. Init failed (data type supposed to be amino-acids)\n");
      break;
    }
  }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Tips_At_One_Site_Generic_Float(char *state, int ns, int state_len, int pos, phydbl *p_lk)
{
  int i;
  int state_int;

  for(i=0;i<ns;i++) p_lk[pos+i] = 0.;

  if(Is_Ambigu(state,GENERIC,state_len)) for(i=0;i<ns;i++) p_lk[pos+i] = 1.;
  else
    {
      char format[6];
      sprintf(format,"%%%dd",state_len);
      if(!sscanf(state,format,&state_int))
    {
      PhyML_Fprintf(stderr,"\n. state='%c'",state);
      PhyML_Fprintf(stderr,"\n. Err in file %s at line %d (function '%s')\n",__FILE__,__LINE__);
      Warn_And_Exit("");
    }
      if(state_int > ns)
    {
      PhyML_Fprintf(stderr,"\n. %s %d cstate: %.2s istate: %d state_len: %d.\n",__FILE__,__LINE__,state,state_int,state_len);
      PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s') \n",__FILE__,__LINE__,__FUNCTION__);
      Warn_And_Exit("");
    }
      p_lk[pos+state_int] = 1.;
      /*       PhyML_Printf("\n. %s %d cstate: %.2s istate: %d state_len: %d ns: %d pos: %d",__FILE__,__LINE__,state,state_int,state_len,ns,pos); */
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Tips_At_One_Site_Generic_Int(char *state, int ns, int state_len, int pos, short int *p_pars)
{
  int i;
  int state_int;

  for(i=0;i<ns;i++) p_pars[pos+i] = 0;

  if(Is_Ambigu(state,GENERIC,state_len)) for(i=0;i<ns;i++) p_pars[pos+i] = 1;
  else
    {
      char format[6];
      sprintf(format,"%%%dd",state_len);
      if(!sscanf(state,format,&state_int))
        {
          PhyML_Fprintf(stderr,"\n. state='%c'",state);
          PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s') \n",__FILE__,__LINE__,__FUNCTION__);
          Warn_And_Exit("");
        }
      if(state_int > ns)
        {
          PhyML_Fprintf(stderr,"\n. %s %d cstate: %.2s istate: %d state_len: %d.\n",__FILE__,__LINE__,state,state_int,state_len);
          PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s') \n",__FILE__,__LINE__,__FUNCTION__);
          Warn_And_Exit("");
        }
      p_pars[pos+state_int] = 1;
/*       PhyML_Printf("\n* %s %d cstate: %.2s istate: %d state_len: %d ns: %d pos: %d",__FILE__,__LINE__,state,state_int,state_len,ns,pos); */
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Get_All_Partial_Lk_Scale(t_tree *tree, t_edge *b_fcus, t_node *a, t_node *d)
{
  Update_Partial_Lk(tree,b_fcus,d);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Post_Order_Lk(t_node *a, t_node *d, t_tree *tree)
{
  int i,dir;

  dir = -1;
    
  if(d->tax) return;
  else
    {
      if(tree->is_mixt_tree)
        {
          MIXT_Post_Order_Lk(a,d,tree);
          return;
        }

      if(tree->n_root != NULL)
        {
          for(i=0;i<3;++i)
            {
              if(d->v[i] != a && !(a == tree->n_root && d->b[i] == tree->e_root))
                Post_Order_Lk(d,d->v[i],tree);
              else dir = i;
            }
        }
      else
        {
          for(i=0;i<3;i++)
            {
              if(d->v[i] != a)
                Post_Order_Lk(d,d->v[i],tree);
              else dir = i;
            }
        }

      if(dir < 0)
        {
          PhyML_Printf("\n. a->num: %d d->num: %d d->v[0]->num: %d d->v[1]->num: %d d->v[2]->num: %d d->b[0]->num: %d d->b[1]->num: %d d->b[2]->num: %d root ? %d e_root ? %d\n",
                       a?a->num:-1,
                       d?d->num:1,
                       d->v[0]?d->v[0]->num:-1,
                       d->v[1]?d->v[1]->num:-1,
                       d->v[2]?d->v[2]->num:-1,
                       d->b[0]?d->b[0]->num:-1,
                       d->b[1]?d->b[1]->num:-1,
                       d->b[2]?d->b[2]->num:-1,
                       tree->n_root?tree->n_root->num:-1,
                       tree->e_root?tree->e_root->num:-1);
          assert(FALSE);
        }

      /* PhyML_Printf("\n. a:%d [%d] d:%d dir:%d [%p %p] [%p %p %p] [%p %p]", */
      /*              a->num, */
      /*              a == tree->n_root, */
      /*              d->num, */
      /*              dir, */
      /*              d->b[dir], */
      /*              tree->e_root, */
      /*              d->b[0],d->b[1],d->b[2], */
      /*              tree->n_root->b[1],tree->n_root->b[2]); */
      
      if(tree->ignore_root == NO && d->b[dir] == tree->e_root)
        {
          if(d == tree->n_root->v[1]) Update_Partial_Lk(tree,tree->n_root->b[1],d);
          else                        Update_Partial_Lk(tree,tree->n_root->b[2],d);
        }
      else
        {
          Update_Partial_Lk(tree,d->b[dir],d);
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Pre_Order_Lk(t_node *a, t_node *d, t_tree *tree)
{
  int i;

  if(d->tax) return;
  else
    {
      if(tree->is_mixt_tree)
        {
          MIXT_Pre_Order_Lk(a,d,tree);
          return;
        }

      if(tree->n_root)
        {
          for(i=0;i<3;++i)
            {
              if(d->v[i] != a && !(a == tree->n_root && d->b[i] == tree->e_root))
                {
                  Update_Partial_Lk(tree,d->b[i],d);
                  Pre_Order_Lk(d,d->v[i],tree);
                }
            }
        }
      else
        {
          for(i=0;i<3;++i)
            {
              if(d->v[i] != a)
                {
                  Update_Partial_Lk(tree,d->b[i],d);
                  Pre_Order_Lk(d,d->v[i],tree);
                }
            }
        }
    }      
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Updates all partial likelihood vectors. Depending on whether 
// both_sides = YES or NO, only 'up' or 'up'&'down' partials will
// be updates
void Update_All_Partial_Lk(t_tree *tree)
{
#if PHYML_MT_LK_RUNTIME
  if(tree->is_mixt_tree == NO &&
     tree->lk_thread_ctx != NULL &&
     omp_get_max_threads() > 1 &&
     PhyML_Can_Use_Update_Partial_Lk_Wavefront(tree) == YES &&
     tree->mod->s_opt->skip_tree_traversal == NO)
    {
      const int npatterns = (int)tree->n_pattern;
      const int ncatg = (int)tree->mod->ras->n_catg;
      const int ns = (int)tree->mod->ns;
      t_phyml_mt_update_all_partial_mode update_mode = PHYML_MT_UPDATE_ALL_PARTIAL_MODE_WAVEFRONT;
      int site_blocks = 1;
      int nthreads;

      if(PhyML_Ensure_Update_All_Partial_Lk_Wavefront_Cache(tree) == YES)
        {
          nthreads = PhyML_MT_Threads_Update_All_Partial_Lk(tree->lk_wavefront_njobs,
                                                            tree->lk_wavefront_nlevels,
                                                            tree->lk_wavefront_max_width,
                                                            npatterns,ncatg,ns,tree,
                                                            &site_blocks);

          if(nthreads > 1)
            {
              update_mode = PhyML_MT_Select_Update_All_Partial_Lk_Mode(tree);

              if(update_mode == PHYML_MT_UPDATE_ALL_PARTIAL_MODE_LEGACY)
                {
                  PhyML_Update_All_Partial_Lk_Legacy_MT(tree,
                                                        tree->lk_wavefront_jobs,
                                                        tree->lk_wavefront_njobs,
                                                        nthreads);
                }
              else
                {
                  PhyML_Update_All_Partial_Lk_Wavefront_MT(tree,
                                                           tree->lk_wavefront_jobs,
                                                           tree->lk_wavefront_level_offsets,
                                                           tree->lk_wavefront_nlevels,
                                                           nthreads,
                                                           site_blocks);
                }
              return;
            }
        }
    }
#endif

  if(tree->n_root)
    {
      if(tree->ignore_root == NO)
        {          
          Post_Order_Lk(tree->n_root,tree->n_root->v[1],tree);
          Post_Order_Lk(tree->n_root,tree->n_root->v[2],tree);
          
          Update_Partial_Lk(tree,tree->n_root->b[1],tree->n_root);
          Update_Partial_Lk(tree,tree->n_root->b[2],tree->n_root);
          
          if(tree->both_sides == YES)
            {
              Pre_Order_Lk(tree->n_root,tree->n_root->v[2],tree);
              Pre_Order_Lk(tree->n_root,tree->n_root->v[1],tree);
            }
        }
      else
        {

          Post_Order_Lk(tree->e_root->rght,tree->e_root->left,tree);
          Post_Order_Lk(tree->e_root->left,tree->e_root->rght,tree);
          
          if(tree->both_sides == YES)
            {
              Pre_Order_Lk(tree->e_root->rght,tree->e_root->left,tree);
              Pre_Order_Lk(tree->e_root->left,tree->e_root->rght,tree);
            }
        }
    }
  else
    {
      Post_Order_Lk(tree->a_nodes[tree->tip_root],tree->a_nodes[tree->tip_root]->v[0],tree);
      if(tree->both_sides == YES)
        Pre_Order_Lk(tree->a_nodes[tree->tip_root],tree->a_nodes[tree->tip_root]->v[0],tree);
    }
}

#if PHYML_MT_LK_RUNTIME && PHYML_OPT_PARTIAL_LK
static void PhyML_Update_Partial_Lk_Team(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx)
{
  assert(ctx != NULL);

  if(b->left == d && b->update_partial_lk_left == NO) return;
  if(b->rght == d && b->update_partial_lk_rght == NO) return;

  if(tree->is_mixt_tree)
    {
      #pragma omp single
      MIXT_Update_Partial_Lk(tree,b,d);
      return;
    }

  if((tree->io->do_alias_subpatt == YES) &&
     (tree->update_alias_subpatt == YES))
    {
      #pragma omp single
      Alias_One_Subpatt((d==b->left)?(b->rght):(b->left),d,tree);
    }

  if(d->tax) return;

#ifdef BEAGLE
  #pragma omp single
  update_beagle_partials(tree,b,d);
#else
  if(tree->mod->use_m4mod == NO)
    {
      if(tree->mod->ns == 4 || tree->mod->ns == 20)
        {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
          AVX_Update_Partial_Lk_Team(tree,b,d,ctx);
          return;
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
          SSE_Update_Partial_Lk_Team(tree,b,d,ctx);
          return;
#else
          Default_Update_Partial_Lk_Team(tree,b,d,ctx);
          return;
#endif
        }
    }

  #pragma omp single
  Update_Partial_Lk_Generic(tree,b,d);
#endif
}
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

static void Pull_Scaling_Factors_Local(int site, phydbl *site_lk_cat, t_edge *b, t_tree *tree)
{
  unsigned int catg;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  phydbl *dst = tree->unscaled_site_lk_cat + (size_t)site * ncatg;
  int sum_scale_left_cat_local[ncatg];
  int sum_scale_rght_cat_local[ncatg];

  if(tree->apply_lk_scaling == NO)
    {
      tree->fact_sum_scale[site] = 0;
      for(catg=0;catg<ncatg;++catg) dst[catg] = site_lk_cat[catg];
      return;
    }

  switch(tree->scaling_method)
    {
    case SCALE_RATE_SPECIFIC:
      {
        int *sum_scale_left_cat,*sum_scale_rght_cat;
        int exponent;
        phydbl max_sum_scale,min_sum_scale;
        phydbl sum,tmp,dum;

        sum_scale_left_cat = sum_scale_left_cat_local;
        sum_scale_rght_cat = sum_scale_rght_cat_local;

        max_sum_scale =   (phydbl)BIG;
        min_sum_scale =  -(phydbl)BIG;

        for(catg=0;catg<ncatg;++catg)
          {
            sum_scale_left_cat[catg] =
              (b->sum_scale_left)?
              (b->sum_scale_left[site*ncatg+catg]):
              (0.0);

            sum_scale_rght_cat[catg] =
              (b->sum_scale_rght)?
              (b->sum_scale_rght[site*ncatg+catg]):
              (0.0);

            sum = sum_scale_left_cat[catg] + sum_scale_rght_cat[catg];
            dum = log(FABS(site_lk_cat[catg]));

            tmp = sum + ((phydbl)LOGBIG - dum)/(phydbl)LOG2;
            if(tmp < max_sum_scale) max_sum_scale = tmp;

            tmp = sum + ((phydbl)LOGSMALL - dum)/(phydbl)LOG2;
            if(tmp > min_sum_scale) min_sum_scale = tmp;
          }

        if(min_sum_scale > max_sum_scale) min_sum_scale = max_sum_scale;

        tree->fact_sum_scale[site] = (int)((max_sum_scale + min_sum_scale) / 2);

        for(catg=0;catg<ncatg;++catg)
          {
            exponent = -(sum_scale_left_cat[catg]+sum_scale_rght_cat[catg])+tree->fact_sum_scale[site];
            Rate_Correction(exponent,site_lk_cat + catg);
          }
        break;
      }

    case SCALE_FAST:
      {
        const int sum_scale_left =
          (b->sum_scale_left)?
          (b->sum_scale_left[site]):
          (0.0);
        const int sum_scale_rght =
          (b->sum_scale_rght)?
          (b->sum_scale_rght[site]):
          (0.0);

        tree->fact_sum_scale[site] = sum_scale_left + sum_scale_rght;
        break;
      }

    default:
      {
        assert(FALSE);
        break;
      }
    }

  for(catg=0;catg<ncatg;++catg) dst[catg] = site_lk_cat[catg];
}

static phydbl Lk_Site_No_Eigen_Local(unsigned int site,
                                     int state, int ambiguity_check,
                                     const phydbl *p_lk_left, const phydbl *p_lk_rght,
                                     const phydbl *Pij_rr, const phydbl *tPij_rr,
                                     t_edge *b, t_tree *tree,
                                     phydbl *site_lk_cat_local,
                                     int *numerical_warning)
{
  phydbl site_lk,log_site_lk;
  const phydbl *pi = tree->mod->e_frq->pi->v;
  const phydbl *site_lk_cat_ptr;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int nsns = ns * ns;
  unsigned int catg;

  if(tree->mod->s_opt->skip_tree_traversal == NO)
    {
      const phydbl *catg_p_lk_left = p_lk_left;
      const phydbl *catg_p_lk_rght = p_lk_rght;
      const phydbl *catg_Pij_rr = Pij_rr;
      const phydbl *catg_tPij_rr = tPij_rr;

      for(catg=0;catg<ncatg;++catg)
        {
          if(ns == 4 || ns == 20)
            {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
              site_lk_cat_local[catg] = AVX_Lk_Core_One_Class_No_Eigen_Lr(catg_p_lk_left,catg_p_lk_rght,catg_Pij_rr,catg_tPij_rr,pi,ns,ambiguity_check,state);
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
              site_lk_cat_local[catg] = SSE_Lk_Core_One_Class_No_Eigen_Lr((phydbl *)catg_p_lk_left,(phydbl *)catg_p_lk_rght,(phydbl *)catg_Pij_rr,(phydbl *)catg_tPij_rr,(phydbl *)pi,ns,ambiguity_check,state);
#else
              site_lk_cat_local[catg] = Lk_Core_One_Class_No_Eigen_Lr((phydbl *)catg_p_lk_left,(phydbl *)catg_p_lk_rght,(phydbl *)catg_Pij_rr,(phydbl *)pi,ns,ambiguity_check,state);
#endif
            }
          else
            {
              site_lk_cat_local[catg] = Lk_Core_One_Class_No_Eigen_Lr((phydbl *)catg_p_lk_left,(phydbl *)catg_p_lk_rght,(phydbl *)catg_Pij_rr,(phydbl *)pi,ns,ambiguity_check,state);
            }

          catg_Pij_rr += nsns;
          catg_tPij_rr += nsns;
          if(b->left->tax == NO) catg_p_lk_left += ns;
          if(b->rght->tax == NO) catg_p_lk_rght += ns;
        }

      Pull_Scaling_Factors_Local((int)site,site_lk_cat_local,b,tree);
    }

  site_lk_cat_ptr = tree->unscaled_site_lk_cat + (size_t)site * ncatg;
  site_lk = .0;
  for(catg=0;catg<ncatg;++catg) site_lk += site_lk_cat_ptr[catg] * tree->mod->ras->gamma_r_proba->v[catg];

  if(tree->mod->ras->invar == YES)
    {
      int num_prec_issue = NO;
      phydbl inv_site_lk = Invariant_Lk(tree->fact_sum_scale[site],site,&num_prec_issue,tree);

      switch(num_prec_issue)
        {
        case YES:
          tree->fact_sum_scale[site] = 0;
          inv_site_lk = Invariant_Lk(0,site,&num_prec_issue,tree);
          site_lk = inv_site_lk * tree->mod->ras->pinvar->v;
          break;

        case NO:
          site_lk = site_lk * (1. - tree->mod->ras->pinvar->v) + inv_site_lk * tree->mod->ras->pinvar->v;
          break;
        }
    }

  if(site_lk < SMALL)
    {
      site_lk = SMALL;
      *numerical_warning = YES;
    }

  log_site_lk = log(site_lk) - (phydbl)LOG2 * tree->fact_sum_scale[site];
  tree->c_lnL_sorted[site] = log_site_lk;
  tree->cur_site_lk[site] = exp(log_site_lk);
  return log_site_lk;
}

phydbl Lk_Site_Eigen_Local(unsigned int site,
                           const phydbl *expl, const phydbl *dot_prod,
                           t_edge *b, t_tree *tree,
                           phydbl *site_lk_cat_local,
                           int *numerical_warning)
{
  phydbl site_lk,log_site_lk;
  const phydbl *site_lk_cat_ptr;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  unsigned int catg;

  if(tree->mod->s_opt->skip_tree_traversal == NO)
    {
      const phydbl *catg_dot_prod = dot_prod;
      const phydbl *catg_expl = expl;

      for(catg=0;catg<ncatg;++catg)
        {
          if(tree->mod->io->datatype == NT || tree->mod->io->datatype == AA)
            {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
              site_lk_cat_local[catg] = AVX_Lk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns);
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
              site_lk_cat_local[catg] = SSE_Lk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns);
#else
              site_lk_cat_local[catg] = Lk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns);
#endif
            }
          else
            {
              site_lk_cat_local[catg] = Lk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns);
            }

          catg_dot_prod += ns;
          if(catg_expl) catg_expl += ns;
        }

      Pull_Scaling_Factors_Local((int)site,site_lk_cat_local,b,tree);
    }

  site_lk_cat_ptr = tree->unscaled_site_lk_cat + (size_t)site * ncatg;
  site_lk = .0;
  for(catg=0;catg<ncatg;++catg) site_lk += site_lk_cat_ptr[catg] * tree->mod->ras->gamma_r_proba->v[catg];

  if(tree->mod->ras->invar == YES)
    {
      int num_prec_issue = NO;
      phydbl inv_site_lk = Invariant_Lk(tree->fact_sum_scale[site],site,&num_prec_issue,tree);

      switch(num_prec_issue)
        {
        case YES:
          tree->fact_sum_scale[site] = 0;
          inv_site_lk = Invariant_Lk(0,site,&num_prec_issue,tree);
          site_lk = inv_site_lk * tree->mod->ras->pinvar->v;
          break;

        case NO:
          site_lk = site_lk * (1. - tree->mod->ras->pinvar->v) + inv_site_lk * tree->mod->ras->pinvar->v;
          break;
        }
    }

  if(site_lk < SMALL)
    {
      site_lk = SMALL;
      *numerical_warning = YES;
    }

  log_site_lk = log(site_lk) - (phydbl)LOG2 * tree->fact_sum_scale[site];
  tree->c_lnL_sorted[site] = log_site_lk;
  tree->cur_site_lk[site] = exp(log_site_lk);
  return log_site_lk;
}

static void Lk_dLk_Site_Eigen_Local(unsigned int site,
                                    const phydbl *expl, const phydbl *dot_prod,
                                    t_edge *b, t_tree *tree,
                                    phydbl *site_lk_cat_local,
                                    int *numerical_warning,
                                    phydbl *lk, phydbl *dlk)
{
  unsigned int catg;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const phydbl *catg_dot_prod = dot_prod;
  const phydbl *catg_expl = expl;

  *lk = *dlk = 0.0;

  if(tree->mod->s_opt->skip_tree_traversal == NO)
    {
      for(catg=0;catg<ncatg;++catg)
        {
          phydbl core_lk,core_dlk;

          if(tree->mod->io->datatype == NT || tree->mod->io->datatype == AA)
            {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
              AVX_Lk_dLk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns,&core_lk,&core_dlk);
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
              SSE_Lk_dLk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns,&core_lk,&core_dlk);
#else
              Lk_dLk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns,&core_lk,&core_dlk);
#endif
            }
          else
            {
              Lk_dLk_Core_One_Class_Eigen_Lr((phydbl *)catg_dot_prod,(phydbl *)(catg_expl ? catg_expl : NULL),ns,&core_lk,&core_dlk);
            }

          site_lk_cat_local[catg] = core_lk;
          *lk  += core_lk  * tree->mod->ras->gamma_r_proba->v[catg];
          *dlk += core_dlk * tree->mod->ras->gamma_r_proba->v[catg];

          catg_dot_prod += ns;
          if(catg_expl) catg_expl += 2*ns;
        }

      Pull_Scaling_Factors_Local((int)site,site_lk_cat_local,b,tree);
    }

  if(tree->mod->ras->invar == YES)
    {
      int num_prec_issue = NO;
      phydbl inv_site_lk = Invariant_Lk(tree->fact_sum_scale[site],site,&num_prec_issue,tree);

      switch(num_prec_issue)
        {
        case YES:
          *lk = inv_site_lk * tree->mod->ras->pinvar->v;
          *dlk = 0.0;
          break;

        case NO:
          *lk = *lk * (1. - tree->mod->ras->pinvar->v) + inv_site_lk * tree->mod->ras->pinvar->v;
          *dlk = *dlk * (1. - tree->mod->ras->pinvar->v);
          break;
        }
    }

  if(*lk < SMALL)
    {
      *lk = SMALL;
      *numerical_warning = YES;
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Lk(t_edge *b, t_tree *tree)
{
  unsigned int br,catg,state,ambiguity_check,site;
  phydbl len,*expl,*dot_prod,*p_lk_left,*p_lk_rght;  

  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int npatterns = tree->n_pattern;
  const unsigned int nsncatg = ns * ncatg;
  const int full_tree_eval = (b == NULL);
  int partial_mt_threads = 1;
  int pmat_mt_threads = 1;
#if PHYML_MT_LK_RUNTIME
  t_phyml_mt_update_all_partial_mode partial_mt_mode = PHYML_MT_UPDATE_ALL_PARTIAL_MODE_WAVEFRONT;
  int partial_mt_site_blocks = 1;
  int lk_mt_threads = 1;
  int eigen_mt_threads = 1;
  int mt_force_st_mode = NO;
  int mt_saved_threads = 1;
#endif

  
  tree->numerical_warning = NO;
  
  /* if(tree->eval_alnL == NO) return UNLIKELY; */
  
  if(b == NULL && tree->mod->s_opt->curr_opt_free_rates == YES)
    {
      tree->mod->s_opt->curr_opt_free_rates = NO;
      Optimize_Free_Rate_Weights(tree,YES,YES);
      tree->mod->s_opt->curr_opt_free_rates = YES;
    }
    
  if(tree->is_mixt_tree == YES) 
    {
#ifdef BEAGLE
      Warn_And_Exit(TODO_BEAGLE);
#endif
      MIXT_Lk(b,tree);
      return tree->c_lnL;
    }
  
  tree->old_lnL = tree->c_lnL;
  

  if(tree->rates && tree->io && tree->io->lk_approx == NORMAL)
    {
#ifdef BEAGLE
      Warn_And_Exit(TODO_BEAGLE);
#endif
      tree->c_lnL = Lk_Normal_Approx(tree);
      return tree->c_lnL;
    }
  
  expl     = tree->expl;
  dot_prod = tree->dot_prod;
 
 if(b == NULL)
   {
     Update_Boundaries(tree->mod);
     Update_RAS(tree->mod);
     Update_Efrq(tree->mod);
     Update_Eigen(tree->mod);
   }

#if PHYML_MT_LK_RUNTIME
  if(tree->lk_thread_ctx != NULL)
    {
      long long pmat_work_est = 0LL;
      long long partial_work_est = 0LL;
      long long eigen_work_est = 0LL;
      const long long site_work_est = (long long)npatterns * (long long)ncatg * (long long)ns;

      lk_mt_threads = PhyML_MT_Threads_Site_Lk((int)npatterns,(int)ncatg,(int)ns,tree);
      if(tree->use_eigen_lr == YES)
        eigen_mt_threads = PhyML_MT_Threads_Update_Eigen_Lr((int)npatterns,(int)ncatg,(int)ns,tree);
      if(tree->use_eigen_lr == NO && tree->mod->s_opt->skip_tree_traversal == NO)
        pmat_mt_threads = PhyML_MT_Threads_Update_PMat(full_tree_eval ? ((2 * tree->n_otu - 3) +
                                                                         ((tree->n_root && tree->ignore_root == NO) ? 2 : 0))
                                                                      : 1,
                                                      (int)ncatg,(int)ns,tree);

      if(tree->mod->s_opt->skip_tree_traversal == NO)
        {
          const int npmat_edges = full_tree_eval ?
            ((2 * tree->n_otu - 3) + ((tree->n_root && tree->ignore_root == NO) ? 2 : 0)) :
            1;

          pmat_work_est = (tree->use_eigen_lr == NO) ?
            ((long long)npmat_edges * (long long)ncatg * (long long)ns * (long long)ns * (long long)ns) :
            0LL;
          eigen_work_est = (tree->use_eigen_lr == YES) ?
            ((long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns) :
            0LL;

          if(full_tree_eval == YES &&
             omp_get_max_threads() > 1 &&
             PhyML_Can_Use_Update_Partial_Lk_Wavefront(tree) == YES &&
             PhyML_Ensure_Update_All_Partial_Lk_Wavefront_Cache(tree) == YES)
            {
              partial_mt_threads = PhyML_MT_Threads_Update_All_Partial_Lk(tree->lk_wavefront_njobs,
                                                                          tree->lk_wavefront_nlevels,
                                                                          tree->lk_wavefront_max_width,
                                                                          (int)npatterns,
                                                                          (int)ncatg,
                                                                          (int)ns,
                                                                          tree,
                                                                          &partial_mt_site_blocks);
              partial_mt_mode = PhyML_MT_Select_Update_All_Partial_Lk_Mode(tree);
              partial_work_est = (long long)tree->lk_wavefront_njobs * (long long)npatterns *
                (long long)ncatg * (long long)ns * (long long)ns;
            }

          if(PhyML_MT_Should_Fallback_ST_Lk(pmat_work_est,
                                            pmat_mt_threads,
                                            partial_work_est,
                                            partial_mt_threads,
                                            eigen_work_est,
                                            eigen_mt_threads,
                                            site_work_est,
                                            lk_mt_threads) == YES)
            {
              partial_mt_threads = 1;
              pmat_mt_threads = 1;
              lk_mt_threads = 1;
              eigen_mt_threads = 1;
              partial_mt_site_blocks = 1;
              mt_force_st_mode = YES;
            }
        }

      mt_saved_threads = PhyML_MT_Enter_Forced_ST_Mode(mt_force_st_mode);
    }
#endif

  
 if(tree->mod->s_opt->skip_tree_traversal == NO)
    {
      if(!b) //Update PMat for all edges
        {
#if PHYML_MT_LK_RUNTIME
          if(tree->lk_thread_ctx != NULL && omp_get_max_threads() > 1)
            {
              const int npmat_edges = (2 * tree->n_otu - 3) +
                ((tree->n_root && tree->ignore_root == NO) ? 2 : 0);
              pmat_mt_threads = PhyML_MT_Threads_Update_PMat(npmat_edges,(int)ncatg,(int)ns,tree);
            }
#endif
          if(pmat_mt_threads > 1)
            {
              #pragma omp parallel for schedule(static) num_threads(pmat_mt_threads)
              for(br=0;br<2*tree->n_otu-3;++br)
                {
                  Update_PMat_At_Given_Edge(tree->a_edges[br],tree);
                }
            }
          else
            {
              for(br=0;br<2*tree->n_otu-3;++br)
                {
                  Update_PMat_At_Given_Edge(tree->a_edges[br],tree);
                }
            }
          
          if(tree->n_root && tree->ignore_root == NO)
            {
              Update_PMat_At_Given_Edge(tree->n_root->b[1],tree);
              Update_PMat_At_Given_Edge(tree->n_root->b[2],tree);
            }
        }
      else //Update PMat for a specific edge
        {
          if(tree->use_eigen_lr == NO)
            {
              if(tree->n_root &&
                 (b == tree->n_root->b[1] || b == tree->n_root->b[2]) &&
                 tree->ignore_root == YES)
                {
                  Update_PMat_At_Given_Edge(tree->e_root,tree);
                }
              else
                {
                  Update_PMat_At_Given_Edge(b,tree);
                }
            }
        }
      
      if(!b)
        {
#if PHYML_MT_LK_RUNTIME
          if(tree->lk_thread_ctx != NULL &&
             omp_get_max_threads() > 1 &&
             PhyML_Can_Use_Update_Partial_Lk_Wavefront(tree) == YES)
            {
              if(PhyML_Ensure_Update_All_Partial_Lk_Wavefront_Cache(tree) == YES)
                {
                  partial_mt_threads = PhyML_MT_Threads_Update_All_Partial_Lk(tree->lk_wavefront_njobs,
                                                                              tree->lk_wavefront_nlevels,
                                                                              tree->lk_wavefront_max_width,
                                                                              (int)npatterns,
                                                                              (int)ncatg,
                                                                              (int)ns,
                                                                              tree,
                                                                              &partial_mt_site_blocks);
                  partial_mt_mode = PhyML_MT_Select_Update_All_Partial_Lk_Mode(tree);
                }
            }
#endif
          if(partial_mt_threads <= 1)
            {
              if(tree->n_root != NULL)
                {
                  if(tree->ignore_root == NO)
                    {
                      Post_Order_Lk(tree->n_root,tree->n_root->v[1],tree);
                      Post_Order_Lk(tree->n_root,tree->n_root->v[2],tree);

                      Update_Partial_Lk(tree,tree->n_root->b[1],tree->n_root);
                      Update_Partial_Lk(tree,tree->n_root->b[2],tree->n_root);

                      if(tree->both_sides == YES)
                        {
                          Pre_Order_Lk(tree->n_root,tree->n_root->v[2],tree);
                          Pre_Order_Lk(tree->n_root,tree->n_root->v[1],tree);
                        }
                    }
                  else
                    {
                      Post_Order_Lk(tree->e_root->rght,tree->e_root->left,tree);
                      Post_Order_Lk(tree->e_root->left,tree->e_root->rght,tree);

                      if(tree->both_sides == YES)
                        {
                          Pre_Order_Lk(tree->e_root->rght,tree->e_root->left,tree);
                          Pre_Order_Lk(tree->e_root->left,tree->e_root->rght,tree);
                        }
                    }
                }
              else
                {
                  Post_Order_Lk(tree->a_nodes[tree->tip_root],tree->a_nodes[tree->tip_root]->v[0],tree);
                  if(tree->both_sides == YES)
                    Pre_Order_Lk(tree->a_nodes[tree->tip_root],tree->a_nodes[tree->tip_root]->v[0],tree);
                }
            }
        }
    }

  if(!b)
    {
      if(tree->n_root) 
        {
          if(tree->ignore_root == NO)
            b = (tree->n_root->v[1]->tax == NO)?(tree->n_root->b[2]):(tree->n_root->b[1]);
          else
            b = tree->e_root;
        }
      else                                        
        b = tree->a_nodes[tree->tip_root]->b[0];
    }

  tree->c_lnL             = .0;
  tree->sum_min_sum_scale = .0;

#if PHYML_MT_LK_RUNTIME
  (void)mt_force_st_mode;
  (void)full_tree_eval;
#endif

#ifdef BEAGLE
  calc_edgelks_beagle(b, tree);
#else

#if PHYML_MT_LK_RUNTIME
  if(tree->lk_wavefront_valid == YES &&
     tree->lk_wavefront_jobs != NULL &&
     tree->lk_wavefront_level_offsets != NULL &&
     partial_mt_threads > 1 &&
     tree->lk_thread_ctx != NULL)
    {
      int pipeline_threads = partial_mt_threads;

      if(eigen_mt_threads > pipeline_threads) pipeline_threads = eigen_mt_threads;
      if(lk_mt_threads > pipeline_threads) pipeline_threads = lk_mt_threads;
      if(partial_mt_mode == PHYML_MT_UPDATE_ALL_PARTIAL_MODE_LEGACY &&
         pipeline_threads > partial_mt_threads)
        pipeline_threads = partial_mt_threads;

      if(pipeline_threads > 1)
        {
          #pragma omp parallel num_threads(pipeline_threads)
            {
              t_lk_thread_ctx *ctx = PhyML_MT_Get_Thread_Ctx(tree);

              assert(ctx != NULL);
              ctx->numerical_warning = NO;

              if(partial_mt_mode == PHYML_MT_UPDATE_ALL_PARTIAL_MODE_LEGACY)
                {
                  PhyML_Update_All_Partial_Lk_Legacy_Team(tree,
                                                          tree->lk_wavefront_jobs,
                                                          tree->lk_wavefront_njobs,
                                                          ctx);
                }
              else
                {
                  PhyML_Update_All_Partial_Lk_Wavefront_Team(tree,
                                                             tree->lk_wavefront_jobs,
                                                             tree->lk_wavefront_level_offsets,
                                                             tree->lk_wavefront_nlevels,
                                                             partial_mt_threads,
                                                             partial_mt_site_blocks,
                                                             ctx);
                }

              if(tree->use_eigen_lr == YES && tree->update_eigen_lr == YES)
                {
                  PhyML_Update_Eigen_And_Lk_Sites_Team(b,tree,expl,ctx);
                }
              else
                {
                  PhyML_Lk_Sites_Team(b,tree,expl,ctx);
                }
            }

          tree->c_lnL = .0;
          for(site=0;site<npatterns;++site)
            {
              if(tree->data->wght[site] > SMALL)
                tree->c_lnL += tree->data->wght[site] * tree->c_lnL_sorted[site];
            }

          tree->numerical_warning = PhyML_MT_Collect_Warnings(tree,pipeline_threads);
          PhyML_MT_Leave_Forced_ST_Mode(mt_saved_threads);
          return tree->c_lnL;
        }
    }
#endif

  
  if(tree->update_eigen_lr == YES)
    {
#if PHYML_MT_LK_RUNTIME
      if(!(lk_mt_threads > 1 && tree->use_eigen_lr == YES))
#endif
        Update_Eigen_Lr(b,tree);
    }
  
  if(tree->use_eigen_lr == YES)
    {  
      for(catg=0;catg<ncatg;++catg)
        {
          len = MAX(0.0,b->l->v)*tree->mod->ras->gamma_rr->v[catg];
          len *= tree->mod->br_len_mult->v;
          if(tree->mixt_tree != NULL)     len *= tree->mixt_tree->mod->ras->gamma_rr->v[tree->mod->ras->parent_class_number];
          if(len < tree->mod->l_min)      len = tree->mod->l_min;
          else if(len > tree->mod->l_max) len = tree->mod->l_max;
          for(state=0;state<ns;++state) expl[catg*ns+state] = exp(tree->mod->eigen->e_val[state]*len);
        }
    }

#if PHYML_MT_LK_RUNTIME
  {
    const int nthreads = lk_mt_threads;

    if(nthreads > 1 && tree->lk_thread_ctx != NULL)
      {
        #pragma omp parallel num_threads(nthreads)
          {
            t_lk_thread_ctx *ctx = PhyML_MT_Get_Thread_Ctx(tree);

            assert(ctx != NULL);
            ctx->numerical_warning = NO;

            if(tree->use_eigen_lr == YES && tree->update_eigen_lr == YES)
              {
                PhyML_Update_Eigen_And_Lk_Sites_Team(b,tree,expl,ctx);
              }
            else
              {
                PhyML_Lk_Sites_Team(b,tree,expl,ctx);
              }
          }

        tree->c_lnL = .0;
        for(site=0;site<npatterns;++site)
          {
            if(tree->data->wght[site] > SMALL)
              tree->c_lnL += tree->data->wght[site] * tree->c_lnL_sorted[site];
          }

        tree->numerical_warning = PhyML_MT_Collect_Warnings(tree,nthreads);
        PhyML_MT_Leave_Forced_ST_Mode(mt_saved_threads);
        return tree->c_lnL;
      }
  }
#endif

  p_lk_left = b->p_lk_left;
  p_lk_rght = b->rght->tax ? b->p_lk_tip_r : b->p_lk_rght;

  for(site=0;site<npatterns;++site)
    {
      ambiguity_check = -1;
      state           = -1;
      tree->curr_site = site;
      
      if((b->rght->tax) && (tree->mod->s_opt->greedy == NO))
        {
          ambiguity_check = b->rght->c_seq->is_ambigu[tree->curr_site];
          if(ambiguity_check == NO)
            {
              state = b->rght->c_seq->d_state[tree->curr_site];
            }
        }
      
      if(tree->mod->use_m4mod) ambiguity_check = YES;
      
      if(tree->use_eigen_lr == YES)
        {
          if(tree->data->wght[site] > SMALL) Lk_Core_Eigen_Lr(expl,dot_prod,b,tree);
          dot_prod += nsncatg;
        }
      else
        {
          if(tree->data->wght[site] > SMALL) Lk_Core(state,ambiguity_check,p_lk_left,p_lk_rght,b->Pij_rr,b->tPij_rr,b,tree);

          if(b->rght->tax == YES)
            {
              p_lk_left += nsncatg;
              p_lk_rght += ns;
            }
          else
            {
              p_lk_left += nsncatg;
              p_lk_rght += nsncatg;
            }
        }
    }
#endif

#if PHYML_MT_LK_RUNTIME
  PhyML_MT_Leave_Forced_ST_Mode(mt_saved_threads);
#endif
  return tree->c_lnL;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
// First derivative of the log-likelihood with respect
// to the length of edge b
phydbl dLk(phydbl *l, t_edge *b, t_tree *tree)
{
  unsigned int catg,state,site;
  phydbl len,rr,var;
  phydbl lk,dlk,dlnlk,lnlk;
  phydbl ev,expevlen;
   
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int npattern = tree->n_pattern;
#if PHYML_MT_LK_RUNTIME
  int dlk_mt_threads = 1;
  int mt_force_st_mode = NO;
  int mt_saved_threads = 1;
#endif

  phydbl *dot_prod = tree->dot_prod;
  phydbl *expl = tree->expl;

  tree->numerical_warning = NO;
  
  assert(isnan(*l) == FALSE);

  if(*l < tree->mod->l_min)      *l = tree->mod->l_min;
  else if(*l > tree->mod->l_max) *l = tree->mod->l_max;      

  assert(b != NULL);
  
  if(tree->is_mixt_tree == YES)
    {
#ifdef BEAGLE
      Warn_And_Exit(TODO_BEAGLE);
#endif
      return MIXT_dLk(l,b,tree);
    }

#if PHYML_MT_LK_RUNTIME
  if(tree->lk_thread_ctx != NULL)
    {
      dlk_mt_threads = PhyML_MT_Threads_dLk((int)npattern,(int)ncatg,(int)ns,tree);
      if(PhyML_MT_Should_Fallback_ST_dLk((tree->update_eigen_lr == YES) ?
                                         ((long long)npattern * (long long)ncatg * (long long)ns * (long long)ns) : 0LL,
                                         dlk_mt_threads,
                                         (long long)npattern * (long long)ncatg * (long long)ns * (long long)ns,
                                         dlk_mt_threads) == YES)
        {
          dlk_mt_threads = 1;
          mt_force_st_mode = YES;
        }

      mt_saved_threads = PhyML_MT_Enter_Forced_ST_Mode(mt_force_st_mode);
    }
#endif
    
  if(tree->update_eigen_lr == YES)
    {
#if PHYML_MT_LK_RUNTIME
      if(dlk_mt_threads <= 1)
#endif
        Update_Eigen_Lr(b,tree);
    }
  
  for(catg=0;catg<ncatg;catg++)
    {
      rr = tree->mod->ras->gamma_rr->v[catg];
      if(tree->mixt_tree) rr = tree->mixt_tree->mod->ras->gamma_rr->v[tree->mod->ras->parent_class_number];
      rr *=  tree->mod->br_len_mult->v;

      len = (*l) * rr;
      /* var = tree->mod->l_var_sigma * rr*rr; */
      var = (*l) * tree->mod->l_var_sigma->v * rr*rr;
      
      if(isinf(len) || isnan(len)) 
        {
          PhyML_Fprintf(stderr,"\n. len=%f rr=%f l=%f",len,rr,*l);
          assert(FALSE);
        }

      if(len < tree->mod->l_min)      len = tree->mod->l_min;
      else if(len > tree->mod->l_max) len = tree->mod->l_max;      
      // value of rr should be corrected too if any of these two conditions
      // is true. Leads to numerical precision issues though...
      
      
      for(state=0;state<ns;++state) 
        {
          ev = tree->mod->eigen->e_val[state];
          expevlen = exp(ev*len);

          if(tree->mod->gamma_mgf_bl == YES)
            {
              expl[catg*2*ns + 2*state]      = POW(1. - ev*var/len,-len*len/var); 
              expl[catg*2*ns + 2*state + 1]  = expl[catg*2*ns + 2*state];
              expl[catg*2*ns + 2*state + 1] *= -(ev * rr/(1.-ev*var/len) + 2.*len*rr*LOG(1.-ev*var/len)/var);              
            }
          else
            {
              expl[catg*2*ns + 2*state]     = expevlen;
              expl[catg*2*ns + 2*state + 1] = expevlen*ev*rr;
            }
        }
    }
    
  dlnlk  = 0.0;
  lnlk   = 0.0;

#if PHYML_MT_LK_RUNTIME
  {
    const int nthreads = dlk_mt_threads;

    if(nthreads > 1 && tree->lk_thread_ctx != NULL)
      {
        #pragma omp parallel num_threads(nthreads)
          {
            unsigned int begin,end;
            unsigned int local_site;
            t_lk_thread_ctx *ctx = PhyML_MT_Get_Thread_Ctx(tree);

            assert(ctx != NULL);
            ctx->numerical_warning = NO;

            if(tree->update_eigen_lr == YES) PhyML_Update_Eigen_Lr_Team(b,tree,ctx);
            #pragma omp barrier

            PhyML_MT_Get_Site_Range(npattern,&begin,&end);

            for(local_site=begin;local_site<end;++local_site)
              {
                if(tree->data->wght[local_site] > SMALL)
                  {
                    phydbl lk_site,dlk_site,log_site_lk;
                    int site_warning = NO;

                    Lk_dLk_Site_Eigen_Local(local_site,
                                            expl,
                                            dot_prod + (size_t)local_site * ns * ncatg,
                                            b,tree,
                                            ctx->site_lk_cat,
                                            &site_warning,
                                            &lk_site,&dlk_site);

                    dlk_site /= lk_site;
                    log_site_lk = log(lk_site) - (phydbl)LOG2 * tree->fact_sum_scale[local_site];
                    tree->site_dlnL[local_site] = tree->data->wght[local_site] * dlk_site;
                    tree->c_lnL_sorted[local_site] = log_site_lk;
                    tree->cur_site_lk[local_site] = exp(log_site_lk);
                    if(site_warning == YES) ctx->numerical_warning = YES;
                  }
                else
                  {
                    tree->site_dlnL[local_site] = .0;
                  }
              }
          }

        dlnlk = .0;
        lnlk  = .0;
        for(site=0;site<npattern;++site)
          {
            if(tree->data->wght[site] > SMALL)
              {
                dlnlk += tree->site_dlnL[site];
                lnlk += tree->data->wght[site] * tree->c_lnL_sorted[site];
              }
          }

        tree->numerical_warning = PhyML_MT_Collect_Warnings(tree,nthreads);
        tree->c_dlnL = dlnlk;
        tree->c_lnL  = lnlk;
        PhyML_MT_Leave_Forced_ST_Mode(mt_saved_threads);
        return tree->c_lnL;
      }
  }
#endif
  
  for(site=0;site<npattern;++site)
    {
      if(tree->data->wght[site] > SMALL) 
        {
          tree->curr_site = site;
          
          Lk_dLk_Core_Eigen_Lr(expl,dot_prod+site*ns*ncatg,b,&lk,&dlk,tree);

          assert(lk > .0);
          
          dlk /= lk;
          dlnlk += tree->data->wght[site] * dlk;
          lnlk += tree->data->wght[site] * (log(lk) - (phydbl)LOG2 * tree->fact_sum_scale[site]);
        }
    }

  tree->c_dlnL = dlnlk;
  tree->c_lnL  = lnlk;

#if PHYML_MT_LK_RUNTIME
  PhyML_MT_Leave_Forced_ST_Mode(mt_saved_threads);
#endif
  return tree->c_lnL;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

/* Core of the likelihood calculation. Assume that the partial likelihoods on both
   sides of t_edge *b are up-to-date. Calculate the log-likelihood at one site.
   Note: this function can be used to evaluate first or second derivative of the 
   likelihood function with respect to the length of b, at a given site. Hence, be
   careful with the meaning of 'site_lk'. If 'derivative=TRUE', then site_lk is 
   either the first or second derivative of the likelihood at that site, given the
   length of edge 'b'.
*/

phydbl Lk_Core(int state, int ambiguity_check,
               phydbl *p_lk_left, phydbl *p_lk_rght,
               phydbl *Pij_rr,
               phydbl *tPij_rr,
               t_edge *b,
               t_tree *tree)
{
  phydbl site_lk,res,*pi,*site_lk_cat,log_site_lk;
  unsigned int catg;
  
  const unsigned int ns    = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int site  = tree->curr_site;  
  const unsigned nsns      = ns*ns;

  
  assert(tree->data->wght[site] > SMALL);
  
  pi = tree->mod->e_frq->pi->v;
  
  if(tree->mod->s_opt->skip_tree_traversal == NO)
    {
      for(catg=0;catg<ncatg;++catg)
        {
          if(ns == 4 || ns == 20)
            {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
              tree->site_lk_cat[catg] = AVX_Lk_Core_One_Class_No_Eigen_Lr(p_lk_left,p_lk_rght,Pij_rr,tPij_rr,pi,ns,ambiguity_check,state);
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
              tree->site_lk_cat[catg] = SSE_Lk_Core_One_Class_No_Eigen_Lr(p_lk_left,p_lk_rght,Pij_rr,tPij_rr,pi,ns,ambiguity_check,state);
#else
              tree->site_lk_cat[catg] = Lk_Core_One_Class_No_Eigen_Lr(p_lk_left,p_lk_rght,Pij_rr,pi,ns,ambiguity_check,state);
#endif
            }
          else
            {
              tree->site_lk_cat[catg] = Lk_Core_One_Class_No_Eigen_Lr(p_lk_left,p_lk_rght,Pij_rr,pi,ns, ambiguity_check, state);
            }
          
          Pij_rr += nsns;
          tPij_rr += nsns;
          if(b->left->tax == NO) p_lk_left += ns;
          if(b->rght->tax == NO) p_lk_rght += ns;
        }

      Pull_Scaling_Factors(site,b,tree);

    }

  site_lk = .0;
  site_lk_cat = tree->unscaled_site_lk_cat + site*ncatg;
  for(catg=0;catg<ncatg;++catg) site_lk += site_lk_cat[catg] * tree->mod->ras->gamma_r_proba->v[catg];

  if(tree->mod->ras->invar == YES)
    {
      int num_prec_issue = NO;
      phydbl inv_site_lk = Invariant_Lk(tree->fact_sum_scale[site],site,&num_prec_issue,tree);

      switch(num_prec_issue)
        {
        case YES :         
          {
            assert(isinf(inv_site_lk));
            tree->fact_sum_scale[site] = 0;
            inv_site_lk = Invariant_Lk(0,site,&num_prec_issue,tree);
            site_lk = inv_site_lk * tree->mod->ras->pinvar->v;
            break;
          }
        case NO : 
          {
            site_lk = site_lk * (1. - tree->mod->ras->pinvar->v) + inv_site_lk * tree->mod->ras->pinvar->v;
            break;
          }
        }

    }

  if(tree->apply_lk_scaling == YES) res = site_lk / pow(2,tree->fact_sum_scale[site]);
  else                              res = site_lk;

  if(site_lk < SMALL)
    {
      site_lk = SMALL;
      tree->numerical_warning = YES;
    }

  
  log_site_lk = log(site_lk) - (phydbl)LOG2 * tree->fact_sum_scale[site]; // log_site_lk =  log(site_lk_scaled / 2^(left_subtree+right_subtree))
  tree->c_lnL_sorted[site] = log_site_lk;
  tree->c_lnL += tree->data->wght[site] * log_site_lk;
  tree->cur_site_lk[site] = exp(log_site_lk); // note to self : add opt out option to avoid calculating this if not necessary
    
  /* printf("\n. clnL: %f",tree->c_lnL); */
  return res;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Lk_Core_Eigen_Lr(phydbl *expl, phydbl *dot_prod, t_edge *b, t_tree *tree)
{
  phydbl site_lk,res,*site_lk_cat,log_site_lk;
  unsigned int catg;
  int num_prec_issue;
  
  const unsigned int ns    = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int site  = tree->curr_site;

  assert(tree->data->wght[site] > SMALL);
  
  if(tree->mod->s_opt->skip_tree_traversal == NO)
    {
      for(catg=0;catg<ncatg;++catg)
        {
          if(tree->mod->io->datatype == NT || tree->mod->io->datatype == AA)
            {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
              tree->site_lk_cat[catg] = AVX_Lk_Core_One_Class_Eigen_Lr(dot_prod,expl ? expl : NULL,ns);
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
              tree->site_lk_cat[catg] = SSE_Lk_Core_One_Class_Eigen_Lr(dot_prod,expl ? expl : NULL,ns);
#else
              tree->site_lk_cat[catg] = Lk_Core_One_Class_Eigen_Lr(dot_prod,expl ? expl : NULL,ns);
#endif
            }
          else
            {
              tree->site_lk_cat[catg] = Lk_Core_One_Class_Eigen_Lr(dot_prod,expl ? expl : NULL,ns);
            }

          dot_prod += ns;
          if(expl) expl += ns;
        }

      Pull_Scaling_Factors(site,b,tree);

    }
  
  
  site_lk_cat = tree->unscaled_site_lk_cat + site*ncatg;
  site_lk = .0;
  for(catg=0;catg<ncatg;++catg) site_lk += site_lk_cat[catg] * tree->mod->ras->gamma_r_proba->v[catg];
  
  if(tree->mod->ras->invar == YES)
    {
      num_prec_issue = NO;
      phydbl inv_site_lk = Invariant_Lk(tree->fact_sum_scale[site],site,&num_prec_issue,tree);

      switch(num_prec_issue)
        {
        case YES :
          {
            assert(isinf(inv_site_lk));
            tree->fact_sum_scale[site] = 0;
            inv_site_lk = Invariant_Lk(0,site,&num_prec_issue,tree);
            site_lk = inv_site_lk * tree->mod->ras->pinvar->v;
            break;
          }
        case NO :
          {
            site_lk = site_lk * (1. - tree->mod->ras->pinvar->v) + inv_site_lk * tree->mod->ras->pinvar->v;
            break;
          }
        }
    }

  if(site_lk < SMALL)
    {
      site_lk = SMALL;
      tree->numerical_warning = YES;
    }

  // likelihood (or 1st, 2nd derivative) not rescaled here. Valid only if all partial likelihoods
  // were scaled using the same factor, i.e., when scaling_method == SCALE_FAST. In this case, the
  // scaling factors will cancel out in dlk/lk and d2lk/lk
  res = site_lk;
  
  log_site_lk = log(site_lk) - (phydbl)LOG2 * tree->fact_sum_scale[site]; // log_site_lk =  log(site_lk_scaled / 2^(left_subtree+right_subtree))
  tree->c_lnL_sorted[site] = log_site_lk;
  tree->c_lnL += tree->data->wght[site] * log_site_lk;
  tree->cur_site_lk[site] = exp(log_site_lk); // note to self : add opt out option to avoid calculating this if not necessary
      
  return res;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
// Compute likelihood and first derivative of likelihood with respect to the length of edge b *unscaled* 
void Lk_dLk_Core_Eigen_Lr(phydbl *expl, phydbl *dot_prod, t_edge *b, phydbl *lk, phydbl *dlk, t_tree *tree)
{
  phydbl core_lk,core_dlk;
  unsigned int catg;
  int num_prec_issue;
  
  const unsigned int ns    = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int site  = tree->curr_site;
  
  *lk = *dlk = 0.0;

  assert(tree->data->wght[site] > SMALL);
  
  if(tree->mod->s_opt->skip_tree_traversal == NO)
    {
      for(catg=0;catg<ncatg;++catg)
        {
          if(tree->mod->io->datatype == NT || tree->mod->io->datatype == AA)
            {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
              AVX_Lk_dLk_Core_One_Class_Eigen_Lr(dot_prod,
                                                 expl ? expl : NULL,
                                                 ns,&core_lk,&core_dlk);
#elif ((defined(__SSE__) || defined(__SSE2__) ||  defined(__SSE3__)) && !defined(DISABLE_NATIVE))
              SSE_Lk_dLk_Core_One_Class_Eigen_Lr(dot_prod,
                                                 expl ? expl : NULL,
                                                 ns,&core_lk,&core_dlk);
#else
              Lk_dLk_Core_One_Class_Eigen_Lr(dot_prod,
                                             expl ? expl : NULL,
                                             ns,&core_lk,&core_dlk);
#endif
            }
          else
            {
              Lk_dLk_Core_One_Class_Eigen_Lr(dot_prod,
                                             expl ? expl : NULL,
                                             ns,&core_lk,&core_dlk);
            }

          *lk  += core_lk * tree->mod->ras->gamma_r_proba->v[catg];
          *dlk += core_dlk * tree->mod->ras->gamma_r_proba->v[catg];
          
          dot_prod += ns;
          if(expl) expl += 2*ns;
        }
      Pull_Scaling_Factors(site,b,tree);
    }
    
  if(tree->mod->ras->invar == YES)
    {
      num_prec_issue = NO;
      phydbl inv_site_lk = Invariant_Lk(tree->fact_sum_scale[site],site,&num_prec_issue,tree);

      switch(num_prec_issue)
        {
        case YES :
          {
            *lk = inv_site_lk * tree->mod->ras->pinvar->v;
            *dlk = 0.0;
            break;
          }
        case NO :
          {
            *lk = *lk * (1. - tree->mod->ras->pinvar->v) + inv_site_lk * tree->mod->ras->pinvar->v;
            *dlk = *dlk * (1. - tree->mod->ras->pinvar->v);
            break;
          }
        }
    }

  if(*lk < SMALL)
    {
      *lk = SMALL;
      tree->numerical_warning = YES;
    }
}


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#if PHYML_MT_LK_RUNTIME
static void Default_Update_Eigen_Lr_Team(t_edge *b, t_tree *tree, t_lk_thread_ctx *ctx)
{
  unsigned int site,catg,i,j;
  phydbl *r_e_vect,*l_e_vect,*pi;
  const unsigned int npattern = tree->n_pattern;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int nsncatg = ns*ncatg;
  phydbl *left_pi;

  assert(ctx != NULL);
  left_pi = ctx->p_lk_left_pi;

  r_e_vect = tree->mod->eigen->r_e_vect;
  l_e_vect = tree->mod->eigen->l_e_vect;
  pi       = tree->mod->e_frq->pi->v;

  #pragma omp for schedule(static)
  for(site=0;site<npattern;++site)
    {
      phydbl *site_dot_prod = tree->dot_prod + (size_t)site * nsncatg;
      const phydbl *site_p_lk_left = b->left->tax ?
        (b->p_lk_tip_l + (size_t)site * ns) :
        (b->p_lk_left + (size_t)site * nsncatg);
      const phydbl *site_p_lk_rght = b->rght->tax ?
        (b->p_lk_tip_r + (size_t)site * ns) :
        (b->p_lk_rght + (size_t)site * nsncatg);

      if(tree->data->wght[site] > SMALL)
        {
          for(catg=0;catg<ncatg;++catg)
            {
              for(j=0;j<ns;++j) left_pi[j] = site_p_lk_left[j] * pi[j];

              for(i=0;i<ns;++i)
                {
                  phydbl left = .0;
                  phydbl rght = .0;

                  for(j=0;j<ns;++j)
                    {
                      left += r_e_vect[j*ns + i] * left_pi[j];
                      rght += l_e_vect[i*ns + j] * site_p_lk_rght[j];
                    }

                  site_dot_prod[i] = left * rght;
                }

              site_dot_prod += ns;
              if(b->left->tax == NO) site_p_lk_left += ns;
              if(b->rght->tax == NO) site_p_lk_rght += ns;
            }
        }
    }
}

static void Default_Update_Eigen_And_Lk_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx)
{
  unsigned int site,catg,i,j;
  phydbl *r_e_vect,*l_e_vect,*pi;
  const unsigned int npattern = tree->n_pattern;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  phydbl *left_pi;
  phydbl *site_dot_prod;

  assert(ctx != NULL);
  left_pi = ctx->p_lk_left_pi;
  site_dot_prod = ctx->site_dot_prod;

  r_e_vect = tree->mod->eigen->r_e_vect;
  l_e_vect = tree->mod->eigen->l_e_vect;
  pi       = tree->mod->e_frq->pi->v;

  #pragma omp for schedule(static)
  for(site=0;site<npattern;++site)
    {
      const phydbl *site_p_lk_left = b->left->tax ?
        (b->p_lk_tip_l + (size_t)site * ns) :
        (b->p_lk_left + (size_t)site * ns * ncatg);
      const phydbl *site_p_lk_rght = b->rght->tax ?
        (b->p_lk_tip_r + (size_t)site * ns) :
        (b->p_lk_rght + (size_t)site * ns * ncatg);

      if(tree->data->wght[site] > SMALL)
        {
          int site_warning = NO;
          phydbl *catg_dot_prod = site_dot_prod;
          phydbl *global_dot_prod = tree->dot_prod + (size_t)site * ns * ncatg;

          for(catg=0;catg<ncatg;++catg)
            {
              for(j=0;j<ns;++j) left_pi[j] = site_p_lk_left[j] * pi[j];

              for(i=0;i<ns;++i)
                {
                  phydbl left = .0;
                  phydbl rght = .0;

                  for(j=0;j<ns;++j)
                    {
                      left += r_e_vect[j*ns + i] * left_pi[j];
                      rght += l_e_vect[i*ns + j] * site_p_lk_rght[j];
                    }

                  catg_dot_prod[i] = left * rght;
                  global_dot_prod[catg*ns + i] = catg_dot_prod[i];
                }

              catg_dot_prod += ns;
              if(b->left->tax == NO) site_p_lk_left += ns;
              if(b->rght->tax == NO) site_p_lk_rght += ns;
            }

          Lk_Site_Eigen_Local(site,expl,site_dot_prod,b,tree,ctx->site_lk_cat,&site_warning);
          if(site_warning == YES) ctx->numerical_warning = YES;
        }
    }
}

static int PhyML_MT_Collect_Warnings(t_tree *tree, int nthreads)
{
  int thread_id;
  int warning = NO;

  for(thread_id=0;thread_id<nthreads && thread_id<tree->lk_mt_max_threads;++thread_id)
    {
      if(tree->lk_thread_ctx[thread_id].numerical_warning == YES)
        {
          warning = YES;
          break;
        }
    }

  return warning;
}

static void PhyML_Lk_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx)
{
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int nsncatg = ns * ncatg;
  unsigned int site;

  assert(ctx != NULL);

  #pragma omp for schedule(static)
  for(site=0;site<tree->n_pattern;++site)
    {
      if(tree->data->wght[site] > SMALL)
        {
          int site_warning = NO;

          if(tree->use_eigen_lr == YES)
            {
              Lk_Site_Eigen_Local(site,
                                  expl,
                                  tree->dot_prod + (size_t)site * nsncatg,
                                  b,tree,
                                  ctx->site_lk_cat,
                                  &site_warning);
            }
          else
            {
              const phydbl *site_p_lk_left = (b->left->tax == YES)?
                (b->p_lk_tip_l + (size_t)site * ns):
                (b->p_lk_left + (size_t)site * nsncatg);
              const phydbl *site_p_lk_rght = (b->rght->tax == YES)?
                (b->p_lk_tip_r + (size_t)site * ns):
                (b->p_lk_rght + (size_t)site * nsncatg);
              int site_ambiguity_check = -1;
              int site_state = -1;

              if((b->rght->tax) && (tree->mod->s_opt->greedy == NO))
                {
                  site_ambiguity_check = b->rght->c_seq->is_ambigu[site];
                  if(site_ambiguity_check == NO) site_state = b->rght->c_seq->d_state[site];
                }

              if(tree->mod->use_m4mod) site_ambiguity_check = YES;

              Lk_Site_No_Eigen_Local(site,
                                     site_state,site_ambiguity_check,
                                     site_p_lk_left,site_p_lk_rght,
                                     b->Pij_rr,b->tPij_rr,
                                     b,tree,
                                     ctx->site_lk_cat,
                                     &site_warning);
            }

          if(site_warning == YES) ctx->numerical_warning = YES;
        }
    }
}

static void PhyML_Update_Eigen_Lr_Team(t_edge *b, t_tree *tree, t_lk_thread_ctx *ctx)
{
  if(tree->mod->ns == 4 || tree->mod->ns == 20)
    {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
      AVX_Update_Eigen_Lr_Team(b,tree,ctx);
      return;
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
      SSE_Update_Eigen_Lr_Team(b,tree,ctx);
      return;
#endif
    }

  Default_Update_Eigen_Lr_Team(b,tree,ctx);
}

static void PhyML_Update_Eigen_And_Lk_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx)
{
  if(tree->mod->ns == 4 || tree->mod->ns == 20)
    {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
      AVX_Update_Eigen_And_Lk_Sites_Team(b,tree,expl,ctx);
      return;
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
      SSE_Update_Eigen_And_Lk_Sites_Team(b,tree,expl,ctx);
      return;
#endif
    }

  Default_Update_Eigen_And_Lk_Sites_Team(b,tree,expl,ctx);
}
#endif

void Update_Eigen_Lr(t_edge *b, t_tree *tree)
{
  unsigned int site,catg,i,j;
  phydbl *dot_prod,*r_e_vect,*l_e_vect,*p_lk_left,*p_lk_rght,*pi;
  phydbl left,rght;
  
  const unsigned int npattern = tree->n_pattern;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int nsncatg = ns*ncatg;

  
  if(tree->is_mixt_tree == YES)
    {
      MIXT_Update_Eigen_Lr(b,tree);
      return;
    }

  if(tree->mod->ns == 4 || tree->mod->ns == 20)
    {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
      AVX_Update_Eigen_Lr(b,tree);
      return;
#elif ((defined(__SSE__) || defined(__SSE2__) ||  defined(__SSE3__)) && !defined(DISABLE_NATIVE))
      SSE_Update_Eigen_Lr(b,tree);
      return;
#endif
    }
    
  assert(tree->update_eigen_lr == YES);
  
  dot_prod = tree->dot_prod;
  r_e_vect = tree->mod->eigen->r_e_vect;
  l_e_vect = tree->mod->eigen->l_e_vect;
  pi       = tree->mod->e_frq->pi->v;
    
  if(b->left->tax == YES) p_lk_left = b->p_lk_tip_l;
  else                    p_lk_left = b->p_lk_left;

  if(b->rght->tax == YES) p_lk_rght = b->p_lk_tip_r;
  else                    p_lk_rght = b->p_lk_rght;

#if PHYML_MT_LK_RUNTIME
  {
    const int nthreads = PhyML_MT_Threads_Update_Eigen_Lr((int)npattern,(int)ncatg,(int)ns,tree);

    if(nthreads > 1 && tree->lk_thread_ctx != NULL)
      {
        #pragma omp parallel num_threads(nthreads)
        {
          t_lk_thread_ctx *ctx = PhyML_MT_Get_Thread_Ctx(tree);
          PhyML_Update_Eigen_Lr_Team(b,tree,ctx);
        }
        return;
      }
  }
#endif

  for(site=0;site<npattern;++site)
    {
      if(tree->data->wght[site] > SMALL)
        {
          for(catg=0;catg<ncatg;++catg)
            {
              for(i=0;i<ns;++i)
                {
                  left = rght = 0.0;
                  for(j=0;j<ns;++j)
                    {
                      left += r_e_vect[j*ns + i] * p_lk_left[j] * pi[j];
                      rght += l_e_vect[i*ns + j] * p_lk_rght[j];
                    }
                  dot_prod[i] = left*rght;
                }
              dot_prod += ns;
              if(b->left->tax == NO) p_lk_left += ns;
              if(b->rght->tax == NO) p_lk_rght += ns;
            }
          if(b->left->tax == YES) p_lk_left += ns;
          if(b->rght->tax == YES) p_lk_rght += ns;
        }
      else
        {
          if(b->left->tax == YES) p_lk_left += ns;
          else p_lk_left += nsncatg;
          
          if(b->rght->tax == YES) p_lk_rght += ns;
          else p_lk_rght += nsncatg;

          dot_prod += nsncatg;
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Rate_Correction(int exponent, phydbl *site_lk_cat)
{
  int piecewise_exponent;
  phydbl multiplier,dum;
  unsigned long long int one = 1;
  
  dum = *site_lk_cat;
  if(exponent >= 0)
    {
      /* Multiply by 2^exponent */
      do
        {
          piecewise_exponent = MIN(exponent,63);
          multiplier = (phydbl)(one << piecewise_exponent);
          dum = dum * multiplier;
          exponent = exponent - piecewise_exponent;
        }
      while(exponent != 0);
    }
  else
    {
      /* Divide by 2^exponent */
      do
        {
          piecewise_exponent = MAX(exponent,-63);
          multiplier = 1. / (phydbl)(one << -piecewise_exponent);
          dum = dum * multiplier;
          exponent = exponent - piecewise_exponent;
        }
      while(exponent != 0);
    }

  *site_lk_cat = dum;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Lk_Core_One_Class_Eigen_Lr(phydbl *dot_prod, phydbl *expl, int ns)
{
  unsigned int l;
  phydbl lk = 0.0;
  if(expl != NULL) for(l=0;l<ns;++l) lk += dot_prod[l] * expl[l]; 
  else for(l=0;l<ns;++l) lk += dot_prod[l];
  return lk;
}


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Lk_dLk_Core_One_Class_Eigen_Lr(phydbl *dot_prod, phydbl *expl, unsigned int ns, phydbl *lk, phydbl *dlk)
{
  unsigned int i;

  *lk = *dlk = 0.0;
  for(i=0;i<ns;++i)
    {
      *lk  += dot_prod[i] * expl[2*i];
      *dlk += dot_prod[i] * expl[2*i+1];      
    }
}
 
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Lk_Core_One_Class_No_Eigen_Lr(phydbl *p_lk_left, phydbl *p_lk_rght, phydbl *Pij, phydbl *pi, int ns, int ambiguity_check, int state)
{
  unsigned int l,k;
  phydbl lk = 0.0;
  phydbl sum;

    
  if(ambiguity_check == NO)/* tip case */
    {      
      sum = .0;
      Pij += state*ns;
      for(l=0;l<ns;++l) sum += Pij[l] * p_lk_left[l];               
      lk += sum * pi[state];
    }
  else /* If the character observed at the tip is ambiguous: ns x ns terms to consider */
    {
      for(k=0;k<ns;++k)
        {
          if(p_lk_rght[k] > .0) /* Only bother ascending into the subtrees if the likelihood of state k, at site "site*dim2" is > 0 */
            {
              sum = .0;
              for(l=0;l<ns;l++)
                {
                  sum += Pij[l] * p_lk_left[l];
                }
              lk += sum * pi[k] * p_lk_rght[k];
            }
          Pij += ns;
        }
    } 

  return lk;
}

/* #endif */

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Returns the scaled likelihood for invariable sites
phydbl Invariant_Lk(int fact_sum_scale, int site, int *num_prec_issue, t_tree *tree)
{
  int exponent,piecewise_exponent;
  phydbl multiplier;
  phydbl inv_site_lk = 0.;
  
  (*num_prec_issue) = NO;
  
  /* The substitution model does include invariable sites */
  if(tree->mod->ras->invar == YES)
    {
      /* The site is invariant */
      if(tree->data->invar[site] > -0.5)
        {
          inv_site_lk = tree->mod->e_frq->pi->v[tree->data->invar[site]];
          
          /* printf("\n. inv_site_lk = %f [%c] [%d] invar: %d",inv_site_lk,tree->data->c_seq[0]->state[site],tree->data->invar[site],tree->data->invar[site]); */

          if(tree->apply_lk_scaling == YES)
            {
              exponent = fact_sum_scale;              
              do
                {
                  piecewise_exponent = MIN(exponent,63);
                  multiplier = (phydbl)((unsigned long long)(1) << piecewise_exponent);
                  inv_site_lk *= multiplier;
                  exponent -= piecewise_exponent;
                }
              while(exponent != 0);
            }

          /* Update the value of site_lk */
          if(isinf(inv_site_lk)) // P(D|r=0) >> P(D|r>0) => assume P(D) = P(D|r=0)P(r=0)
            {
              int i;
              PhyML_Fprintf(stderr,"\n. fact_sum_scale: %d",fact_sum_scale);              
              PhyML_Fprintf(stderr,"\n. pi: %f",tree->mod->e_frq->pi->v[tree->data->invar[site]]);              
              for(i=0;i<tree->mod->ns;i++) PhyML_Fprintf(stderr,"\n. pi %d: %f",i,tree->mod->e_frq->pi->v[i]);
              PhyML_Fprintf(stderr,"\n. Numerical precision issue alert.");
              PhyML_Fprintf(stderr,"\n. File %s at line %d (function '%s')\n",__FILE__,__LINE__,__FUNCTION__);
              (*num_prec_issue) = YES;
            }
        }
    }

  return inv_site_lk;

}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

/* Update partial likelihood on edge b on the side of b where
   node d lies.
*/

void Update_Partial_Lk(t_tree *tree, t_edge *b, t_node *d)
{
  /* if(tree->eval_alnL == NO) return; */
  if(b->left == d && b->update_partial_lk_left == NO) return;
  if(b->rght == d && b->update_partial_lk_rght == NO) return;
  
  if(tree->is_mixt_tree)
    {
      MIXT_Update_Partial_Lk(tree,b,d);
      return;
    }
  
  if((tree->io->do_alias_subpatt == YES) &&
     (tree->update_alias_subpatt == YES))
    Alias_One_Subpatt((d==b->left)?(b->rght):(b->left),d,tree);
  if(d->tax) return;

  
#ifdef BEAGLE
  update_beagle_partials(tree, b, d);
#else
  if(tree->mod->use_m4mod == NO)
    {
      if(tree->mod->ns == 4 || tree->mod->ns == 20)
        {
#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))
          AVX_Update_Partial_Lk(tree,b,d);
#elif ((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)) && !defined(DISABLE_NATIVE))
          SSE_Update_Partial_Lk(tree,b,d);
#else
          Default_Update_Partial_Lk(tree,b,d);
#endif
        }
      else
        {
          Update_Partial_Lk_Generic(tree,b,d);
        }
    }
  else
    {
      Update_Partial_Lk_Generic(tree,b,d);
    }
#endif
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#ifndef BEAGLE

void Update_Partial_Lk_Generic(t_tree *tree, t_edge *b, t_node *d)
{
/*
           |
           |<- b
           |
           d
          / \
         /   \
        /     \
    n_v1   n_v2
*/
  t_node *n_v1, *n_v2;
  phydbl p1_lk1,p2_lk2;
  phydbl *p_lk,*p_lk_v1,*p_lk_v2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int sum_scale_v1_val, sum_scale_v2_val;
  int i,j;
  int catg,site;
  int n_patterns;
  short int ambiguity_check_v1,ambiguity_check_v2;
  int state_v1,state_v2;
  phydbl smallest_p_lk,largest_p_lk;
  int *p_lk_loc;

  unsigned const int ncatg = tree->mod->ras->n_catg;
  unsigned const int ns = tree->mod->ns;
  unsigned const int ncatgns = ncatg * ns;
  unsigned const int nsns = ns * ns;
  

  
  if(tree->n_root && tree->ignore_root == YES &&
     (d == tree->n_root->v[1] || d == tree->n_root->v[2]) &&
     (b == tree->n_root->b[1] || b == tree->n_root->b[2]))
    {
      assert(FALSE);
    }


  state_v1 = state_v2 = -1;
  ambiguity_check_v1 = ambiguity_check_v2 = NO;
  sum_scale_v1_val = sum_scale_v2_val = 0;
  p1_lk1 = p2_lk2 = .0;

  if(d->tax)
    {
      PhyML_Fprintf(stderr,"\n. t_node %d is a leaf...",d->num);
      PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s').\n",__FILE__,__LINE__,__FUNCTION__);
      Exit("\n");
    }

  n_patterns = tree->n_pattern;

  n_v1 = n_v2                 = NULL;
  p_lk = p_lk_v1 = p_lk_v2    = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  sum_scale_v1 = sum_scale_v2 = NULL;
  p_lk_loc                    = NULL;
  smallest_p_lk               = BIG;

  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &p_lk,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&p_lk_v1,&sum_scale_v1,
                     &Pij2,&tPij2,&p_lk_v2,&sum_scale_v2,
                     d,b,tree);

  /* For every site in the alignment */
  for(site=0;site<n_patterns;site++)
    {
      if(tree->data->wght[site] > SMALL)
        {
          state_v1 = state_v2 = -1;
          ambiguity_check_v1 = ambiguity_check_v2 = NO;

          if(tree->mod->s_opt->greedy == NO)
            {
              /* n_v1 and n_v2 are tip nodes */
              if(n_v1 && n_v1->tax)
                {
                  /* Is the state at this tip ambiguous? */
                  ambiguity_check_v1 = n_v1->c_seq->is_ambigu[site];
                  /* if(ambiguity_check_v1 == NO) state_v1 = Get_State_From_Partial_Pars(n_v1->b[0]->p_lk_tip_r,site*ns,tree); */
                  if(ambiguity_check_v1 == NO) state_v1 = n_v1->c_seq->d_state[site];
                }
              
              if(n_v2 && n_v2->tax)
                {
                  /* Is the state at this tip ambiguous? */
                  ambiguity_check_v2 = n_v2->c_seq->is_ambigu[site];
                  /* ambiguity_check_v2 = tree->data->c_seq[n_v2->num]->is_ambigu[site]; */
                  /* if(ambiguity_check_v2 == NO) state_v2 = Get_State_From_Partial_Pars(n_v2->b[0]->p_lk_tip_r,site*ns,tree); */
                  if(ambiguity_check_v2 == NO) state_v2 = n_v2->c_seq->d_state[site];
                }
            }
          
          if(tree->mod->use_m4mod)
            {
              ambiguity_check_v1 = YES;
              ambiguity_check_v2 = YES;
            }
          
          /* For all the rate classes */
          for(catg=0;catg<ncatg;catg++)
            {
              if(tree->mod->ras->skip_rate_cat[catg] == YES) continue;
              
              smallest_p_lk = BIG;
              
              /* For all the states at node d */
              for(i=0;i<tree->mod->ns;i++)
                {
                  p1_lk1 = .0;
                  
                  if(n_v1)
                    {
                      /* n_v1 is a tip */
                      if((n_v1->tax) && (!tree->mod->s_opt->greedy))
                        {
                          if(ambiguity_check_v1 == NO)
                            {
                              /* For the (non-ambiguous) state at node n_v1 */
                              p1_lk1 = Pij1[catg*nsns+i*ns+state_v1];
                            }
                          else
                            {
                              /* For all the states at node n_v1 */
                              for(j=0;j<tree->mod->ns;j++)
                                {
                                  p1_lk1 += Pij1[catg*nsns+i*ns+j] * (phydbl)n_v1->b[0]->p_lk_tip_r[site*ns+j];
                                }
                            }
                        }
                      /* n_v1 is an internal node */
                      else
                        {
                          /* For the states at node n_v1 */
                          for(j=0;j<tree->mod->ns;j++)
                            {
                              p1_lk1 += Pij1[catg*nsns+i*ns+j] * p_lk_v1[site*ncatgns+catg*ns+j];
                            }
                        }
                    }
                  else
                    {
                      p1_lk1 = 1.0;
                    }
                  
                  p2_lk2 = .0;
                  
                  /* We do exactly the same as for node n_v1 but for node n_v2 this time.*/
                  if(n_v2)
                    {
                      /* n_v2 is a tip */
                      if((n_v2->tax) && (!tree->mod->s_opt->greedy))
                        {
                          if(ambiguity_check_v2 == NO)
                            {
                              /* For the (non-ambiguous) state at node n_v2 */
                              p2_lk2 = Pij2[catg*nsns+i*ns+state_v2];
                            }
                          else
                            {
                              /* For all the states at node n_v2 */
                              for(j=0;j<tree->mod->ns;j++)
                                {
                                  p2_lk2 += Pij2[catg*nsns+i*ns+j] * (phydbl)n_v2->b[0]->p_lk_tip_r[site*ns+j];
                                }
                            }
                        }
                      /* n_v2 is an internal node */
                      else
                        {
                          /* For all the states at node n_v2 */
                          for(j=0;j<tree->mod->ns;j++)
                            {
                              p2_lk2 += Pij2[catg*nsns+i*ns+j] * p_lk_v2[site*ncatgns+catg*ns+j];
                            }
                        }
                    }
                  else
                    {
                      p2_lk2 = 1.0;
                    }
                  
                  p_lk[site*ncatgns+catg*ns+i] = p1_lk1 * p2_lk2;
                  
                  /* if(site == 0) PhyML_Printf("\n+ site: %d %G",site,p_lk[site*ncatgns+catg*ns+i]); */
                  
                }
              
              if(tree->scaling_method == SCALE_RATE_SPECIFIC)
                {
                  smallest_p_lk = BIG;
                  for(i=0;i<ns;++i)
                    if(p_lk[site*ncatgns+catg*ns+i] < smallest_p_lk)
                      smallest_p_lk = p_lk[site*ncatgns+catg*ns+i];
                  
                  /* Current scaling values at that site */
                  sum_scale_v1_val = (sum_scale_v1)?(sum_scale_v1[site*ncatg+catg]):(0);
                  sum_scale_v2_val = (sum_scale_v2)?(sum_scale_v2[site*ncatg+catg]):(0);
                  
                  sum_scale[site*ncatg+catg] = sum_scale_v1_val + sum_scale_v2_val;
                  
                  /* Scaling. We have p_lk_lim_inf = 2^-500. Consider for instance that 
                     smallest_p_lk = 2^-600, then curr_scaler_pow will be equal to 100, and
                     each element in the partial likelihood vector will be multiplied by
                     2^100. */
                  if(smallest_p_lk < (phydbl)P_LK_LIM_INF &&
                     tree->mod->augmented == NO &&
                     tree->apply_lk_scaling == YES &&
                     (n_v1->tax == NO || n_v2->tax == NO))
                    
                    {
                      int curr_scaler_pow;
                      curr_scaler_pow = (int)(-500.*LOG2-log(smallest_p_lk))/LOG2;
                      sum_scale[site*ncatg+catg] += curr_scaler_pow;
                      for(i=0;i<ns;++i) Rate_Correction(curr_scaler_pow, p_lk + site*nsns + catg*ns + i);
                      
                    }
                }
            }
          
          if(tree->scaling_method == SCALE_FAST)
            {
              sum_scale_v1_val = (sum_scale_v1)?(sum_scale_v1[site]):(0);
              sum_scale_v2_val = (sum_scale_v2)?(sum_scale_v2[site]):(0);              
              sum_scale[site] = sum_scale_v1_val + sum_scale_v2_val;
              
              sum_scale[site*ncatg+catg] = sum_scale_v1_val + sum_scale_v2_val;
              
              assert(sum_scale[site] < 1024);

              largest_p_lk = -BIG; 
              for(i=0;i<ns*ncatg;++i)
                if(p_lk[site*ncatgns+i] > largest_p_lk)
                  largest_p_lk = p_lk[site*ncatgns+i] ;
              
              if(largest_p_lk < INV_TWO_TO_THE_LARGE &&
                 tree->mod->augmented == NO &&
                 tree->apply_lk_scaling == YES)
                {
                  for(i=0;i<ns*ncatg;++i) p_lk[site*ncatgns + i] *= TWO_TO_THE_LARGE;
                  sum_scale[site] += LARGE;
                }              
            }     
        }
      else
        {
          for(i=0;i<ns*ncatg;++i) p_lk[site*ncatgns + i] = 0.0;
        }     
    }
}

void Default_Update_Partial_Lk(t_tree *tree, t_edge *b, t_node *d)
{
/*
           |
           |<- b
           |
           d
          / \
         /   \
        /     \
    n_v1   n_v2
*/
  t_node *n_v1, *n_v2;//d's "left" and "right" neighbor nodes
  phydbl *p_lk;
  phydbl *p_lk_v1,*p_lk_v2;//Partial likelihood vector of node d, d's "left" neighbor, d's "right" neighbor. We fill *p_lk, and assume *p_lk_v1 and *p_lk_v2 are already filled.
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;//Suppose site j, of a certain subtree, has "A" on one tip, and "C" on the other. If you come across this pattern again at site i<j, then you can simply copy the partial likelihoods
  
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int ns = tree->mod->ns;
  const unsigned int n_patterns = tree->n_pattern;


  if(tree->n_root && tree->ignore_root == YES &&
     (d == tree->n_root->v[1] || d == tree->n_root->v[2]) &&
     (b == tree->n_root->b[1] || b == tree->n_root->b[2]))
    {
      assert(FALSE);
    }
  
  if(d->tax)
    {
      PhyML_Fprintf(stderr,"\n. t_node %d is a leaf...",d->num);
      PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s')\n",__FILE__,__LINE__,__FUNCTION__);
      Exit("\n");
    }


  n_v1 = n_v2                 = NULL;
  p_lk = p_lk_v1 = p_lk_v2    = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  p_lk_loc                    = NULL;
  sum_scale_v1                = NULL;
  sum_scale_v2                = NULL;
  
  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &p_lk,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&p_lk_v1,&sum_scale_v1,
                     &Pij2,&tPij2,&p_lk_v2,&sum_scale_v2,
                     d,b,tree);
  
  Core_Default_Update_Partial_Lk(n_v1,n_v2,
                                 p_lk,p_lk_v1,p_lk_v2,
                                 Pij1,Pij2,
                                 sum_scale,sum_scale_v1,sum_scale_v2,
                                 ns,ncatg,n_patterns,
                                 tree->apply_lk_scaling,
                                 tree->data->wght);
}

#if PHYML_MT_LK_RUNTIME && PHYML_OPT_PARTIAL_LK
static void Default_Update_Partial_Lk_Team(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx)
{
  t_node *n_v1, *n_v2;
  phydbl *p_lk;
  phydbl *p_lk_v1,*p_lk_v2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int ns = tree->mod->ns;
  const unsigned int n_patterns = tree->n_pattern;

  (void)ctx;

  n_v1 = n_v2                 = NULL;
  p_lk = p_lk_v1 = p_lk_v2    = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  p_lk_loc                    = NULL;
  sum_scale_v1                = NULL;
  sum_scale_v2                = NULL;

  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &p_lk,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&p_lk_v1,&sum_scale_v1,
                     &Pij2,&tPij2,&p_lk_v2,&sum_scale_v2,
                     d,b,tree);

  Core_Default_Update_Partial_Lk_Team(n_v1,n_v2,
                                      p_lk,p_lk_v1,p_lk_v2,
                                      Pij1,Pij2,
                                      sum_scale,sum_scale_v1,sum_scale_v2,
                                      (int)ns,(int)ncatg,(int)n_patterns,
                                      tree->apply_lk_scaling,
                                      tree->data->wght);
}

static void Default_Update_Partial_Lk_Range(t_tree *tree, t_edge *b, t_node *d,
                                            unsigned int site_begin, unsigned int site_end)
{
  t_node *n_v1, *n_v2;
  phydbl *p_lk;
  phydbl *p_lk_v1,*p_lk_v2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int ns = tree->mod->ns;

  if(site_begin >= site_end) return;

  n_v1 = n_v2                 = NULL;
  p_lk = p_lk_v1 = p_lk_v2    = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  p_lk_loc                    = NULL;
  sum_scale_v1                = NULL;
  sum_scale_v2                = NULL;

  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &p_lk,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&p_lk_v1,&sum_scale_v1,
                     &Pij2,&tPij2,&p_lk_v2,&sum_scale_v2,
                     d,b,tree);

  Core_Default_Update_Partial_Lk_Range(n_v1,n_v2,
                                       p_lk,p_lk_v1,p_lk_v2,
                                       Pij1,Pij2,
                                       sum_scale,sum_scale_v1,sum_scale_v2,
                                       (int)ns,(int)ncatg,
                                       site_begin,site_end,
                                       tree->apply_lk_scaling,
                                       tree->data->wght);
}
#endif


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#if PHYML_MT_LK_RUNTIME && PHYML_OPT_PARTIAL_LK
static void Core_Default_Update_Partial_Lk_Range(const t_node *n_v1, const t_node *n_v2,
                                                 phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                 const phydbl *Pij1, const phydbl *Pij2,
                                                 int *sum_scale0, const int *sum_scale1, const int *sum_scale2,
                                                 const int ns, const int ncatg,
                                                 unsigned int site_begin, unsigned int site_end,
                                                 const int apply_scaling, const phydbl *wght)
{
  const unsigned int ncatgns = (unsigned int)ncatg * (unsigned int)ns;
  const unsigned int nsns = (unsigned int)ns * (unsigned int)ns;
  const int tax_v1 = (n_v1->tax != 0);
  const int tax_v2 = (n_v2->tax != 0);
  const int use_ns4 = (ns == 4);
  const int use_ns20 = (ns == 20);
  const int do_scaling = (apply_scaling == YES);
  const int plk1_catg_step = (tax_v1) ? 0 : ns;
  const int plk2_catg_step = (tax_v2) ? 0 : ns;
  const int plk1_site_stride = (tax_v1) ? ns : (int)ncatgns;
  const int plk2_site_stride = (tax_v2) ? ns : (int)ncatgns;
  const short int *is_ambigu_v1 = (tax_v1) ? n_v1->c_seq->is_ambigu : NULL;
  const short int *is_ambigu_v2 = (tax_v2) ? n_v2->c_seq->is_ambigu : NULL;
  const short int *d_state_v1 = (tax_v1) ? n_v1->c_seq->d_state : NULL;
  const short int *d_state_v2 = (tax_v2) ? n_v2->c_seq->d_state : NULL;
  const phydbl *init_Pij1 = Pij1;
  const phydbl *init_Pij2 = Pij2;

  for(unsigned int site=site_begin;site<site_end;++site)
    {
      unsigned int i,catg;
      int state_v1,state_v2;
      int ambiguity_check_v1,ambiguity_check_v2;
      int sum_scale_v1_val, sum_scale_v2_val;
      phydbl largest_p_lk = -BIG;
      phydbl catg_largest_p_lk;
      phydbl *site_plk0;
      const phydbl *site_plk1,*site_plk2;
      const phydbl *site_Pij1,*site_Pij2;
      phydbl *catg_plk0;
      const phydbl *catg_plk1,*catg_plk2;

      if(wght[site] <= SMALL) continue;

      site_plk0 = plk0 + (size_t)site * ncatgns;
      site_plk1 = plk1 + (size_t)site * plk1_site_stride;
      site_plk2 = plk2 + (size_t)site * plk2_site_stride;

      state_v1 = state_v2 = -1;
      ambiguity_check_v1 = ambiguity_check_v2 = YES;

      if(tax_v1)
        {
          ambiguity_check_v1 = is_ambigu_v1[site];
          if(ambiguity_check_v1 == NO) state_v1 = d_state_v1[site];
        }

      if(tax_v2)
        {
          ambiguity_check_v2 = is_ambigu_v2[site];
          if(ambiguity_check_v2 == NO) state_v2 = d_state_v2[site];
        }

      site_Pij1 = init_Pij1;
      site_Pij2 = init_Pij2;

      for(catg=0;catg<(unsigned int)ncatg;++catg)
        {
          catg_plk0 = site_plk0 + catg * ns;
          catg_plk1 = site_plk1 + catg * plk1_catg_step;
          catg_plk2 = site_plk2 + catg * plk2_catg_step;

          if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exex_4_Max(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exex_4(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exex_20_Max(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exex_20(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Exex(site_Pij1,state_v1,site_Pij2,state_v2,ns,catg_plk0);
                  if(do_scaling)
                    for(i=0;i<(unsigned int)ns;++i)
                      if(catg_plk0[i] > largest_p_lk) largest_p_lk = catg_plk0[i];
                }
            }
          else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_4_Max(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_4(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_20_Max(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_20(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Exin(site_Pij2,state_v2,site_Pij1,catg_plk1,ns,catg_plk0);
                  if(do_scaling)
                    for(i=0;i<(unsigned int)ns;++i)
                      if(catg_plk0[i] > largest_p_lk) largest_p_lk = catg_plk0[i];
                }
            }
          else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_4_Max(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_4(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_20_Max(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_20(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Exin(site_Pij1,state_v1,site_Pij2,catg_plk2,ns,catg_plk0);
                  if(do_scaling)
                    for(i=0;i<(unsigned int)ns;++i)
                      if(catg_plk0[i] > largest_p_lk) largest_p_lk = catg_plk0[i];
                }
            }
          else
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Inin_4_Max(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Inin_4(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Inin_20_Max(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Inin_20(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Inin(site_Pij1,catg_plk1,site_Pij2,catg_plk2,ns,catg_plk0);
                  if(do_scaling)
                    for(i=0;i<(unsigned int)ns;++i)
                      if(catg_plk0[i] > largest_p_lk) largest_p_lk = catg_plk0[i];
                }
            }

          site_Pij1 += nsns;
          site_Pij2 += nsns;
        }

      sum_scale_v1_val = (sum_scale1)?(sum_scale1[site]):(0);
      sum_scale_v2_val = (sum_scale2)?(sum_scale2[site]):(0);
      sum_scale0[site] = sum_scale_v1_val + sum_scale_v2_val;

      if(do_scaling && largest_p_lk < INV_TWO_TO_THE_LARGE)
        {
          for(i=0;i<ncatgns;++i) site_plk0[i] *= TWO_TO_THE_LARGE;
          sum_scale0[site] += LARGE;
        }
    }
}

static void Core_Default_Update_Partial_Lk_Team(const t_node *n_v1, const t_node *n_v2,
                                                phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                const phydbl *Pij1, const phydbl *Pij2,
                                                int *sum_scale0, const int *sum_scale1, const int *sum_scale2,
                                                const int ns, const int ncatg, const int npatterns, const int apply_scaling,
                                                const phydbl *wght)
{
  const unsigned int ncatgns = (unsigned int)ncatg * (unsigned int)ns;
  const unsigned int nsns = (unsigned int)ns * (unsigned int)ns;
  const int tax_v1 = (n_v1->tax != 0);
  const int tax_v2 = (n_v2->tax != 0);
  const int use_ns4 = (ns == 4);
  const int use_ns20 = (ns == 20);
  const int do_scaling = (apply_scaling == YES);
  const int plk1_catg_step = (tax_v1) ? 0 : ns;
  const int plk2_catg_step = (tax_v2) ? 0 : ns;
  const int plk1_site_stride = (tax_v1) ? ns : (int)ncatgns;
  const int plk2_site_stride = (tax_v2) ? ns : (int)ncatgns;
  const short int *is_ambigu_v1 = (tax_v1) ? n_v1->c_seq->is_ambigu : NULL;
  const short int *is_ambigu_v2 = (tax_v2) ? n_v2->c_seq->is_ambigu : NULL;
  const short int *d_state_v1 = (tax_v1) ? n_v1->c_seq->d_state : NULL;
  const short int *d_state_v2 = (tax_v2) ? n_v2->c_seq->d_state : NULL;
  const phydbl *init_Pij1 = Pij1;
  const phydbl *init_Pij2 = Pij2;
  int site;

  #pragma omp for schedule(static)
  for(site=0;site<npatterns;++site)
    {
      unsigned int i,catg;
      int state_v1,state_v2;
      int ambiguity_check_v1,ambiguity_check_v2;
      int sum_scale_v1_val, sum_scale_v2_val;
      phydbl largest_p_lk = -BIG;
      phydbl catg_largest_p_lk;
      phydbl *site_plk0;
      const phydbl *site_plk1,*site_plk2;
      const phydbl *site_Pij1,*site_Pij2;
      phydbl *catg_plk0;
      const phydbl *catg_plk1,*catg_plk2;

      if(wght[site] <= SMALL) continue;

      site_plk0 = plk0 + (size_t)site * ncatgns;
      site_plk1 = plk1 + (size_t)site * plk1_site_stride;
      site_plk2 = plk2 + (size_t)site * plk2_site_stride;

      state_v1 = state_v2 = -1;
      ambiguity_check_v1 = ambiguity_check_v2 = YES;

      if(tax_v1)
        {
          ambiguity_check_v1 = is_ambigu_v1[site];
          if(ambiguity_check_v1 == NO) state_v1 = d_state_v1[site];
        }

      if(tax_v2)
        {
          ambiguity_check_v2 = is_ambigu_v2[site];
          if(ambiguity_check_v2 == NO) state_v2 = d_state_v2[site];
        }

      site_Pij1 = init_Pij1;
      site_Pij2 = init_Pij2;

      for(catg=0;catg<(unsigned int)ncatg;++catg)
        {
          catg_plk0 = site_plk0 + catg * ns;
          catg_plk1 = site_plk1 + catg * plk1_catg_step;
          catg_plk2 = site_plk2 + catg * plk2_catg_step;

          if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exex_4_Max(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exex_4(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exex_20_Max(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exex_20(site_Pij1,state_v1,site_Pij2,state_v2,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Exex(site_Pij1,state_v1,site_Pij2,state_v2,ns,catg_plk0);
                  if(do_scaling)
                    {
                      for(i=0;i<(unsigned int)ns;++i)
                        if(catg_plk0[i] > largest_p_lk)
                          largest_p_lk = catg_plk0[i];
                    }
                }
            }
          else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_4_Max(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_4(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_20_Max(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_20(site_Pij2,state_v2,site_Pij1,catg_plk1,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Exin(site_Pij2,state_v2,site_Pij1,catg_plk1,ns,catg_plk0);
                  if(do_scaling)
                    {
                      for(i=0;i<(unsigned int)ns;++i)
                        if(catg_plk0[i] > largest_p_lk)
                          largest_p_lk = catg_plk0[i];
                    }
                }
            }
          else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_4_Max(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_4(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Exin_20_Max(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Exin_20(site_Pij1,state_v1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Exin(site_Pij1,state_v1,site_Pij2,catg_plk2,ns,catg_plk0);
                  if(do_scaling)
                    {
                      for(i=0;i<(unsigned int)ns;++i)
                        if(catg_plk0[i] > largest_p_lk)
                          largest_p_lk = catg_plk0[i];
                    }
                }
            }
          else
            {
              if(use_ns4)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Inin_4_Max(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Inin_4(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else if(use_ns20)
                {
                  if(do_scaling)
                    {
                      catg_largest_p_lk = Partial_Lk_Inin_20_Max(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                      if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                    }
                  else
                    {
                      Partial_Lk_Inin_20(site_Pij1,catg_plk1,site_Pij2,catg_plk2,catg_plk0);
                    }
                }
              else
                {
                  Partial_Lk_Inin(site_Pij1,catg_plk1,site_Pij2,catg_plk2,ns,catg_plk0);
                  if(do_scaling)
                    {
                      for(i=0;i<(unsigned int)ns;++i)
                        if(catg_plk0[i] > largest_p_lk)
                          largest_p_lk = catg_plk0[i];
                    }
                }
            }

          site_Pij1 += nsns;
          site_Pij2 += nsns;
        }

      sum_scale_v1_val = (sum_scale1)?(sum_scale1[site]):(0);
      sum_scale_v2_val = (sum_scale2)?(sum_scale2[site]):(0);
      sum_scale0[site] = sum_scale_v1_val + sum_scale_v2_val;

      if(do_scaling && largest_p_lk < INV_TWO_TO_THE_LARGE)
        {
          for(i=0;i<ncatgns;++i) site_plk0[i] *= TWO_TO_THE_LARGE;
          sum_scale0[site] += LARGE;
        }
    }
}

static void Core_Default_Update_Partial_Lk_MT(const t_node *n_v1, const t_node *n_v2,
                                              phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                              const phydbl *Pij1, const phydbl *Pij2,
                                              int *sum_scale0, const int *sum_scale1, const int *sum_scale2,
                                              const int ns, const int ncatg, const int npatterns, const int apply_scaling,
                                              const phydbl *wght, const int nthreads)
{
  #pragma omp parallel num_threads(nthreads)
    {
      Core_Default_Update_Partial_Lk_Team(n_v1,n_v2,
                                          plk0,plk1,plk2,
                                          Pij1,Pij2,
                                          sum_scale0,sum_scale1,sum_scale2,
                                          ns,ncatg,npatterns,apply_scaling,
                                          wght);
    }
}
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Core_Default_Update_Partial_Lk(const t_node *n_v1, const t_node *n_v2,
                                    phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                    const phydbl *Pij1, const phydbl *Pij2,
                                    int *sum_scale0, const int *sum_scale1, const int *sum_scale2,
                                    const int ns, const int ncatg, const int npatterns, const int apply_scaling,
                                    const phydbl *wght)
{
#if PHYML_OPT_PARTIAL_LK
#if PHYML_MT_LK_RUNTIME
  {
    const int nthreads = PhyML_MT_Threads_Update_Partial_Lk(npatterns,ncatg,ns);

    if(nthreads > 1)
    {
      Core_Default_Update_Partial_Lk_MT(n_v1,n_v2,
                                        plk0,plk1,plk2,
                                        Pij1,Pij2,
                                        sum_scale0,sum_scale1,sum_scale2,
                                        ns,ncatg,npatterns,apply_scaling,
                                        wght,nthreads);
      return;
    }
  }
#endif
  unsigned int i,site,ncatgns,catg,nsns;
  int state_v1,state_v2;
  int ambiguity_check_v1,ambiguity_check_v2;
  int sum_scale_v1_val, sum_scale_v2_val;
  phydbl largest_p_lk = -BIG;
  phydbl catg_largest_p_lk;
  const phydbl *init_Pij1, *init_Pij2;
  const int tax_v1 = (n_v1->tax != 0);
  const int tax_v2 = (n_v2->tax != 0);
  const int use_ns4 = (ns == 4);
  const int use_ns20 = (ns == 20);
  const int do_scaling = (apply_scaling == YES);
  const int plk1_catg_step = (tax_v1) ? 0 : ns;
  const int plk2_catg_step = (tax_v2) ? 0 : ns;
  const int plk1_site_step = (tax_v1) ? ns : 0;
  const int plk2_site_step = (tax_v2) ? ns : 0;
  const int plk1_zero_wght_step = (tax_v1) ? ns : (ncatg * ns);
  const int plk2_zero_wght_step = (tax_v2) ? ns : (ncatg * ns);
  const short int *is_ambigu_v1 = (tax_v1) ? n_v1->c_seq->is_ambigu : NULL;
  const short int *is_ambigu_v2 = (tax_v2) ? n_v2->c_seq->is_ambigu : NULL;
  const short int *d_state_v1 = (tax_v1) ? n_v1->c_seq->d_state : NULL;
  const short int *d_state_v2 = (tax_v2) ? n_v2->c_seq->d_state : NULL;
  
  ncatgns = ncatg*ns;
  nsns = ns*ns;
  init_Pij1 = Pij1;
  init_Pij2 = Pij2;
  
  /* For every site in the alignment */
  for(site=0;site<npatterns;site++)
    {
      if(wght[site] > SMALL)
        {
          state_v1 = state_v2 = -1;
          ambiguity_check_v1 = ambiguity_check_v2 = YES;
          
          if(tax_v1)
            {
              ambiguity_check_v1 = is_ambigu_v1[site];
              if(ambiguity_check_v1 == NO) state_v1 = d_state_v1[site];
            }
          
          if(tax_v2)
            {
              ambiguity_check_v2 = is_ambigu_v2[site];
              if(ambiguity_check_v2 == NO) state_v2 = d_state_v2[site];
            }
          
          Pij1 = init_Pij1;
          Pij2 = init_Pij2;
          if(do_scaling) largest_p_lk = -BIG;
          
          /* The ambiguity case is site-specific, not catg-specific. */
          if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
            {
              for(catg=0;catg<ncatg;++catg)
                {
                  if(use_ns4)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Exex_4_Max(Pij1,state_v1,Pij2,state_v2,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Exex_4(Pij1,state_v1,Pij2,state_v2,plk0);
                        }
                    }
                  else if(use_ns20)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Exex_20_Max(Pij1,state_v1,Pij2,state_v2,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Exex_20(Pij1,state_v1,Pij2,state_v2,plk0);
                        }
                    }
                  else
                    {
                      Partial_Lk_Exex(Pij1,state_v1,Pij2,state_v2,ns,plk0);
                      if(do_scaling)
                        {
                          for(i=0;i<ns;++i)
                            if(plk0[i] > largest_p_lk)
                              largest_p_lk = plk0[i];
                        }
                    }
                  Pij1 += nsns;
                  Pij2 += nsns;
                  plk1 += plk1_catg_step;
                  plk2 += plk2_catg_step;
                  plk0 += ns;
                }
            }
          else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
            {
              for(catg=0;catg<ncatg;++catg)
                {
                  if(use_ns4)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Exin_4_Max(Pij2,state_v2,Pij1,plk1,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Exin_4(Pij2,state_v2,Pij1,plk1,plk0);
                        }
                    }
                  else if(use_ns20)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Exin_20_Max(Pij2,state_v2,Pij1,plk1,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Exin_20(Pij2,state_v2,Pij1,plk1,plk0);
                        }
                    }
                  else
                    {
                      Partial_Lk_Exin(Pij2,state_v2,Pij1,plk1,ns,plk0);
                      if(do_scaling)
                        {
                          for(i=0;i<ns;++i)
                            if(plk0[i] > largest_p_lk)
                              largest_p_lk = plk0[i];
                        }
                    }
                  Pij1 += nsns;
                  Pij2 += nsns;
                  plk1 += plk1_catg_step;
                  plk2 += plk2_catg_step;
                  plk0 += ns;
                }
            }
          else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
            {
              for(catg=0;catg<ncatg;++catg)
                {
                  if(use_ns4)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Exin_4_Max(Pij1,state_v1,Pij2,plk2,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Exin_4(Pij1,state_v1,Pij2,plk2,plk0);
                        }
                    }
                  else if(use_ns20)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Exin_20_Max(Pij1,state_v1,Pij2,plk2,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Exin_20(Pij1,state_v1,Pij2,plk2,plk0);
                        }
                    }
                  else
                    {
                      Partial_Lk_Exin(Pij1,state_v1,Pij2,plk2,ns,plk0);
                      if(do_scaling)
                        {
                          for(i=0;i<ns;++i)
                            if(plk0[i] > largest_p_lk)
                              largest_p_lk = plk0[i];
                        }
                    }
                  Pij1 += nsns;
                  Pij2 += nsns;
                  plk1 += plk1_catg_step;
                  plk2 += plk2_catg_step;
                  plk0 += ns;
                }
            }
          else
            {
              for(catg=0;catg<ncatg;++catg)
                {
                  if(use_ns4)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Inin_4_Max(Pij1,plk1,Pij2,plk2,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Inin_4(Pij1,plk1,Pij2,plk2,plk0);
                        }
                    }
                  else if(use_ns20)
                    {
                      if(do_scaling)
                        {
                          catg_largest_p_lk = Partial_Lk_Inin_20_Max(Pij1,plk1,Pij2,plk2,plk0);
                          if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                        }
                      else
                        {
                          Partial_Lk_Inin_20(Pij1,plk1,Pij2,plk2,plk0);
                        }
                    }
                  else
                    {
                      Partial_Lk_Inin(Pij1,plk1,Pij2,plk2,ns,plk0);
                      if(do_scaling)
                        {
                          for(i=0;i<ns;++i)
                            if(plk0[i] > largest_p_lk)
                              largest_p_lk = plk0[i];
                        }
                    }
                  Pij1 += nsns;
                  Pij2 += nsns;
                  plk1 += plk1_catg_step;
                  plk2 += plk2_catg_step;
                  plk0 += ns;
                }
            }
          
          plk1 += plk1_site_step;
          plk2 += plk2_site_step;
          
          sum_scale_v1_val = (sum_scale1)?(sum_scale1[site]):(0);
          sum_scale_v2_val = (sum_scale2)?(sum_scale2[site]):(0);
          sum_scale0[site] = sum_scale_v1_val + sum_scale_v2_val;
          
          if(do_scaling && largest_p_lk < INV_TWO_TO_THE_LARGE)
            {
              plk0 -= ncatgns;
              for(i=0;i<ncatgns;++i) plk0[i] *= TWO_TO_THE_LARGE;
              sum_scale0[site] += LARGE;
              plk0 += ncatgns;
            }
        }
      else
        {
          plk0 += ncatgns;
          plk1 += plk1_zero_wght_step;
          plk2 += plk2_zero_wght_step;
        }
    }
#else
  unsigned int i,site,ncatgns,catg,nsns;
  int state_v1,state_v2;
  int ambiguity_check_v1,ambiguity_check_v2;
  int sum_scale_v1_val, sum_scale_v2_val;
  phydbl largest_p_lk;
  const phydbl *init_Pij1, *init_Pij2;
  
  ncatgns = ncatg*ns;
  nsns = ns*ns;
  init_Pij1 = Pij1;
  init_Pij2 = Pij2;
  
  /* For every site in the alignment */
  for(site=0;site<npatterns;site++)
    {
      if(wght[site] > SMALL)
        {
          state_v1 = state_v2 = -1;
          ambiguity_check_v1 = ambiguity_check_v2 = YES;
          
          /* n_v1 and n_v2 are tip nodes */
          if(n_v1->tax)
            {
              /* Is the state at this tip ambiguous? */
              ambiguity_check_v1 = n_v1->c_seq->is_ambigu[site];
              if(ambiguity_check_v1 == NO) state_v1 = n_v1->c_seq->d_state[site];
            }
          
          if(n_v2->tax)
            {
              /* Is the state at this tip ambiguous? */
              ambiguity_check_v2 = n_v2->c_seq->is_ambigu[site];
              if(ambiguity_check_v2 == NO) state_v2 = n_v2->c_seq->d_state[site];
            }
          
          Pij1 = init_Pij1;
          Pij2 = init_Pij2;
          
          /* For all the rate classes */
          for(catg=0;catg<ncatg;++catg)
            {
              if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
                {
                  Partial_Lk_Exex(Pij1,state_v1,
                                  Pij2,state_v2,
                                  ns,plk0);
                }
              else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
                {
                  Partial_Lk_Exin(Pij2,state_v2,
                                  Pij1,plk1,
                                  ns,plk0);
                }
              else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
                {
                  Partial_Lk_Exin(Pij1,state_v1,
                                  Pij2,plk2,
                                  ns,plk0);
                }
              else
                {
                  Partial_Lk_Inin(Pij1,plk1,
                                  Pij2,plk2,
                                  ns,plk0);
                }
              
              Pij1 += nsns;
              Pij2 += nsns;
              
              plk1 += (n_v1->tax) ? 0 : ns;
              plk2 += (n_v2->tax) ? 0 : ns;
              plk0 += ns;
            }
          
          plk1 += (n_v1->tax) ? ns : 0;
          plk2 += (n_v2->tax) ? ns : 0;
          
          sum_scale_v1_val = (sum_scale1)?(sum_scale1[site]):(0);
          sum_scale_v2_val = (sum_scale2)?(sum_scale2[site]):(0);
          sum_scale0[site] = sum_scale_v1_val + sum_scale_v2_val;
          
          plk0 -= ncatgns;
          largest_p_lk = -BIG;
          for(i=0;i<ncatgns;++i)
            if(plk0[i] > largest_p_lk)
              largest_p_lk = plk0[i];
          
          if(largest_p_lk < INV_TWO_TO_THE_LARGE && apply_scaling == YES)
            {
              for(i=0;i<ncatgns;++i) plk0[i] *= TWO_TO_THE_LARGE;
              sum_scale0[site] += LARGE;
            }
          plk0 += ncatgns;
        }
      else
        {
          plk0 += ncatgns;
          plk1 += (n_v1->tax) ? ns : ncatgns;
          plk2 += (n_v2->tax) ? ns : ncatgns;          
        }
    }
#endif
}
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Return_Abs_Lk(t_tree *tree)
{
  Lk(NULL,tree);
  return FABS(tree->c_lnL);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

matrix *ML_Dist(calign *data, t_mod *mod)
{
  int i,j,k,l;
  phydbl init;
  int n_catg;
  phydbl d_max,sum;
  matrix *mat;
  calign *twodata,*tmpdata;
  int state0, state1;
  phydbl *F,len;
  eigen *eigen_struct;

  tmpdata         = (calign *)mCalloc(1,sizeof(calign));
  tmpdata->c_seq  = (align **)mCalloc(2,sizeof(align *));
  tmpdata->obs_state_frq  = (phydbl *)mCalloc(mod->ns,sizeof(phydbl));
  tmpdata->ambigu = (short int *)mCalloc(data->crunch_len,sizeof(short int));
  F               = (phydbl *)mCalloc(mod->ns*mod->ns,sizeof(phydbl ));
  eigen_struct    = (eigen *)Make_Eigen_Struct(mod->ns);

  Set_Update_Eigen(YES,mod);
  Update_Boundaries(mod);
  Update_Eigen(mod);

  tmpdata->n_otu = 2;

  tmpdata->crunch_len = data->crunch_len;
  tmpdata->init_len   = data->init_len;

  mat = NULL;
  if(mod->io->datatype == NT)           mat = (mod->whichmodel < 10)?(K80_dist(data,1E+6)):(JC69_Dist(data,mod));
  else if(mod->io->datatype == AA)      mat = JC69_Dist(data,mod);
  else if(mod->io->datatype == GENERIC) mat = JC69_Dist(data,mod);

  
  for(i=0;i<mod->ras->n_catg;i++) /* Don't use the discrete gamma distribution */
    {
      mod->ras->gamma_rr->v[i]      = 1.0;
      mod->ras->gamma_r_proba->v[i] = 1.0;
    }

  n_catg = mod->ras->n_catg;
  mod->ras->n_catg = 1;

  for(j=0;j<data->n_otu-1;j++)
    {
      tmpdata->c_seq[0]       = data->c_seq[j];
      tmpdata->c_seq[0]->name = data->c_seq[j]->name;
      tmpdata->wght           = data->wght;

      for(k=j+1;k<data->n_otu;k++)
        {
          tmpdata->c_seq[1]       = data->c_seq[k];
          tmpdata->c_seq[1]->name = data->c_seq[k]->name;
          
          twodata = Compact_Cdata(tmpdata,mod->io);
          
          Check_Ambiguities(twodata,mod->io->datatype,mod->io->state_len);
          
          Hide_Ambiguities(twodata);
          
          init = mat->dist[j][k];
          
          if((init > DIST_MAX-SMALL) || (init < .0)) init = 0.1;
          
          d_max = init;
          
          for(i=0;i<mod->ns*mod->ns;++i) F[i]=.0;
          len = 0.0;
          for(l=0;l<twodata->c_seq[0]->len;++l)
            {
              state0 = Assign_State(twodata->c_seq[0]->state+l*mod->io->state_len,mod->io->datatype,mod->io->state_len);
              state1 = Assign_State(twodata->c_seq[1]->state+l*mod->io->state_len,mod->io->datatype,mod->io->state_len);
              
              if((state0 > -1) && (state1 > -1))
                {
                  F[mod->ns*state0+state1] += twodata->wght[l];
                  len += twodata->wght[l];
                }
            }
          
          if(len > .0) 
            {
              for(i=0;i<mod->ns*mod->ns;++i) F[i] /= len;
            }
          
          sum = 0.;
          for(i=0;i<mod->ns*mod->ns;++i) sum += F[i];
          
          /* for(i=0;i<mod->ns*mod->ns;++i) PhyML_Printf("\n. %g",F[i]); */

          /* if(sum < .001) d_max = -1.; */
          if(sum < .001) d_max = init;
          else if((sum > 1. - .001) && (sum < 1. + .001)) Opt_Dist_F(&(d_max),F,mod);
          else
            {
              PhyML_Fprintf(stderr,"\n\n. sum = %f",sum);
              PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s') \n",__FILE__,__LINE__,__FUNCTION__);
              Exit("");
            }
          
          if(d_max >= DIST_MAX) d_max = DIST_MAX;
          
          
          /* Do not correct for dist < BL_MIN, otherwise Fill_Missing_Dist
           *  will not be called
           */
          mat->dist[j][k] = d_max;
          mat->dist[k][j] = mat->dist[j][k];
          Free_Calign(twodata);
        }
    }
  
  mod->ras->n_catg = n_catg;
  
  
  Free(tmpdata->ambigu);
  Free(tmpdata->obs_state_frq);
  Free(tmpdata->c_seq);
  free(tmpdata);
  Free_Eigen(eigen_struct);
  Free(F);

  return mat;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Lk_Given_Two_Seq(calign *data, int numseq1, int numseq2, phydbl dist, t_mod *mod, phydbl *loglk)
{
  align *seq1,*seq2;
  phydbl site_lk,log_site_lk;
  int i,j,k,l;
/*   phydbl **p_lk_l,**p_lk_r; */
  phydbl *p_lk_l,*p_lk_r;
  phydbl len;
  int dim1,dim2;

  dim1 = mod->ns;
  dim2 = mod->ns * mod->ns;

  DiscreteGamma(mod->ras->gamma_r_proba->v, mod->ras->gamma_rr->v, mod->ras->alpha->v,
        mod->ras->alpha->v,mod->ras->n_catg,mod->ras->gamma_median);

  seq1 = data->c_seq[numseq1];
  seq2 = data->c_seq[numseq2];


  p_lk_l = (phydbl *)mCalloc(data->c_seq[0]->len * mod->ns,sizeof(phydbl));
  p_lk_r = (phydbl *)mCalloc(data->c_seq[0]->len * mod->ns,sizeof(phydbl));


  for(i=0;i<mod->ras->n_catg;i++)
    {
      len = dist*mod->ras->gamma_rr->v[i];
      if(len < mod->l_min) len = mod->l_min;
      else if(len > mod->l_max) len = mod->l_max;
      PMat(len,mod,dim2*i,mod->Pij_rr->v,NULL);
    }
  
  if(mod->io->datatype == NT)
    {
      For(i,data->c_seq[0]->len)
        {
          Init_Tips_At_One_Site_Nucleotides_Float(seq1->state[i],i*mod->ns,p_lk_l);
          Init_Tips_At_One_Site_Nucleotides_Float(seq2->state[i],i*mod->ns,p_lk_r);
        }
    }
  else if(mod->io->datatype == AA)
    {
      For(i,data->c_seq[0]->len)
        {
          Init_Tips_At_One_Site_AA_Float(seq1->state[i],i*mod->ns,p_lk_l);
          Init_Tips_At_One_Site_AA_Float(seq2->state[i],i*mod->ns,p_lk_r);
        }
    }
  else
    {
      PhyML_Fprintf(stderr,"\n\n. Not implemented yet...");
      PhyML_Fprintf(stderr,"\n. Err in file %s at line %d\n\n",__FILE__,__LINE__);
      Warn_And_Exit("\n");
    }
  
  
  site_lk = .0;
  *loglk = 0;
  
  For(i,data->c_seq[0]->len)
    {
      if(data->wght[i] > 0.0)
        {
          site_lk = log_site_lk = .0;
          if(!data->ambigu[i])
            {
              for(k=0;k<mod->ns;k++) {if(p_lk_l[i*mod->ns+k] > .0001) break;}
              for(l=0;l<mod->ns;l++) {if(p_lk_r[i*mod->ns+l] > .0001) break;}
              for(j=0;j<mod->ras->n_catg;j++)
                {
                  site_lk +=
                    mod->ras->gamma_r_proba->v[j] *
                    mod->e_frq->pi->v[k] *
                    p_lk_l[i*dim1+k] *
                    mod->Pij_rr->v[j*dim2+k*dim1+l] *
                    p_lk_r[i*dim1+l];
                }
            }
          else
            {
              for(j=0;j<mod->ras->n_catg;j++)
                {
                  for(k=0;k<mod->ns;k++) /*sort sum terms ? No global effect*/
                    {
                      for(l=0;l<mod->ns;l++)
                        {
                          site_lk +=
                            mod->ras->gamma_r_proba->v[j] *
                            mod->e_frq->pi->v[k] *
                            p_lk_l[i*dim1+k] *
                            mod->Pij_rr->v[j*dim2+k*dim1+l] *
                            p_lk_r[i*dim1+l];
                        }
                    }
                }
            }
          
          if(site_lk <= .0)
            {
              PhyML_Fprintf(stderr,"\n\n. '%c' '%c'\n",seq1->state[i],seq2->state[i]);
              Exit("\n. Err: site lk <= 0\n");
            }
          
          log_site_lk += (phydbl)log(site_lk);
          
          *loglk += data->wght[i] * log_site_lk;/* sort sum terms ? No global effect*/
    }
    }

/*   For(i,data->c_seq[0]->len) */
/*     { */
/*       Free(p_lk_l[i]); */
/*       Free(p_lk_r[i]); */
/*     } */

  Free(p_lk_l); Free(p_lk_r);
  return *loglk;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
// Multinomial log likelihood
void Unconstraint_Lk(t_tree *tree)
{
  int i;

  tree->unconstraint_lk = .0;
  for(i=0;i<tree->data->crunch_len;i++) tree->unconstraint_lk += tree->data->wght[i]*(phydbl)log(tree->data->wght[i]);
  tree->unconstraint_lk -= tree->data->init_len*(phydbl)log(tree->data->init_len);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
// Log-likelihood assuming a tree with edge of infinite lengths
void Composite_Lk(t_tree *tree)
{
  int i;
  tree->composite_lk = 0.0;
  for(i=0;i<tree->mod->ns;++i)
    tree->composite_lk +=
      tree->data->obs_state_frq[i]*
      tree->data->init_len*
      tree->n_otu*
      log(tree->mod->e_frq->pi->v[i]);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Partial_Lk_Tips_Double(t_tree *tree)
{
  unsigned int curr_site,i,dim1;

  if(tree->is_mixt_tree == YES) return;
  
  dim1 = tree->mod->ns;


  for(i=0;i<tree->n_otu;i++)
    {
      if(!tree->a_nodes[i]->c_seq || 
	 strcmp(tree->a_nodes[i]->c_seq->name,tree->a_nodes[i]->name))
        {
          PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s') \n",__FILE__,__LINE__,__FUNCTION__);
          Exit("");
        }
    }

  for(curr_site=0;curr_site<tree->data->crunch_len;curr_site++)
    {
      for(i=0;i<tree->n_otu;i++)
        {          
          if (tree->io->datatype == NT)
            Init_Tips_At_One_Site_Nucleotides_Float(tree->a_nodes[i]->c_seq->state[curr_site],
                                                    curr_site*dim1,
                                                    tree->a_nodes[i]->b[0]->p_lk_tip_r);
          else if(tree->io->datatype == AA)
            Init_Tips_At_One_Site_AA_Float(tree->a_nodes[i]->c_seq->state[curr_site],
                                           curr_site*dim1,
                                           tree->a_nodes[i]->b[0]->p_lk_tip_r);
          
          else if(tree->io->datatype == GENERIC)
            Init_Tips_At_One_Site_Generic_Float(tree->a_nodes[i]->c_seq->state+curr_site*tree->mod->io->state_len,
                                                tree->mod->ns,
                                                tree->mod->io->state_len,
                                                curr_site*dim1,
                                                tree->a_nodes[i]->b[0]->p_lk_tip_r);
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Partial_Lk_Tips_Int(t_tree *tree)
{
  int curr_site,i,dim1;

  if(tree->is_mixt_tree == YES) return;

  dim1 = tree->mod->ns;

  for(i=0;i<tree->n_otu;i++)
    {
      if(!tree->a_nodes[i]->c_seq || 
	 strcmp(tree->a_nodes[i]->c_seq->name,tree->a_nodes[i]->name))
        {
          PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d (function '%s') \n",__FILE__,__LINE__,__FUNCTION__);
          Exit("");
        }
    }

  for(curr_site=0;curr_site<tree->data->crunch_len;curr_site++)
    {
      for(i=0;i<tree->n_otu;i++)
        {
          /* printf("\n. site: %3d %c",curr_site,tree->a_nodes[i]->c_seq->state[curr_site]); */
          /* printf("\n. init at %s %p",tree->a_nodes[i]->name,tree->a_nodes[i]->b[0]->p_lk_tip_r); fflush(NULL); */
          if(tree->io->datatype == NT)
            {
              Init_Tips_At_One_Site_Nucleotides_Float(tree->a_nodes[i]->c_seq->state[curr_site],
                                                    curr_site*dim1,
                                                    tree->a_nodes[i]->b[0]->p_lk_tip_r);
              /* Init_Tips_At_One_Site_Nucleotides_Int(tree->data->c_seq[i]->state[curr_site], */
              /* 					    curr_site*dim1, */
              /* 					    tree->a_nodes[i]->b[0]->p_lk_tip_r); */
            }
          else if(tree->io->datatype == AA)
            Init_Tips_At_One_Site_AA_Float(tree->a_nodes[i]->c_seq->state[curr_site],
                                           curr_site*dim1,
                                           tree->a_nodes[i]->b[0]->p_lk_tip_r);
          /* Init_Tips_At_One_Site_AA_Int(tree->data->c_seq[i]->state[curr_site], */
          /* 				 curr_site*dim1,					    */
          /* 				 tree->a_nodes[i]->b[0]->p_lk_tip_r); */
          
          else if(tree->io->datatype == GENERIC)
            {
              Init_Tips_At_One_Site_Generic_Float(tree->a_nodes[i]->c_seq->state+curr_site*tree->mod->io->state_len,
                                                  tree->mod->ns,
                                                  tree->mod->io->state_len,
                                                  curr_site*dim1,
                                                  tree->a_nodes[i]->b[0]->p_lk_tip_r);
              
              /* Init_Tips_At_One_Site_Generic_Int(tree->data->c_seq[i]->state+curr_site*tree->mod->io->state_len, */
              /* 					tree->mod->ns, */
              /* 					tree->mod->io->state_len, */
              /* 					curr_site*dim1, */
              /* 					tree->a_nodes[i]->b[0]->p_lk_tip_r); */
            }
#ifdef BEAGLE
          //Recall that tip partials are stored on the branch leading
          //to the tip, rather than on the tip itself (hence `p_lk_tip_idx`
          //is a field of the branch (i.e. b[0]) rather than the node.
          //Secondly, the BEAGLE's partial buffers are laid out as
          //BEAGLE's partials buffer = [ tax1, tax2, ..., taxN, b1Left, b2Left, b3Left,...,bMLeft, b1Rght, b2Rght, b3Rght,...,bMRght] (N taxa, M branches)
        tree->a_nodes[i]->b[0]->p_lk_tip_idx = i;
#endif
      }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Update_PMat_At_Given_Edge(t_edge *b_fcus, t_tree *tree)
{
  int i;
  phydbl len;
  phydbl l_min, l_max;
  phydbl mean, var;
  
  assert(b_fcus);
  assert(tree);
  assert(tree->eval_alnL == YES);
  
  if(tree->is_mixt_tree == YES)
    {
      MIXT_Update_PMat_At_Given_Edge(b_fcus,tree);
      return;
    }
  
  if(b_fcus->Pij_rr == NULL)
    {
      PhyML_Printf("\n. b_fcus is e_root ? %d node left: %d node rght: %d left is root ? %d right is root ? %d [%p] [%d] [%d]",
                   (b_fcus == tree->e_root) ? 1 : 0,
                   b_fcus->left->num,
                   b_fcus->rght->num,
                   (b_fcus->left == tree->n_root) ? 1 : 0,
                   (b_fcus->rght == tree->n_root) ? 1 : 0,
                   tree->aux_tree,
                   tree->eval_alnL,
                   tree->is_mixt_tree);
      assert(false);
    }

  if(tree->mixt_tree != NULL) assert(tree->mod->ras->n_catg == 1);
  
  if(tree->mod->gamma_mgf_bl == YES) Set_Br_Len_Var(b_fcus,tree);

  l_min = tree->mod->l_min;
  l_max = tree->mod->l_max;

  len = -1.0;

  if(tree->mod->log_l == YES) b_fcus->l->v = exp(b_fcus->l->v);

  for(i=0;i<tree->mod->ras->n_catg;i++)
    {
      if(tree->mod->ras->skip_rate_cat[i] == YES) continue;
      
      //Update the branch length
      if(b_fcus->has_zero_br_len == YES)
        {
#ifdef BEAGLE
          Warn_And_Exit(TODO_BEAGLE);
#endif
          len = -1.0;
          mean = -1.0;
          var  = -1.0;
        }
      else
        {
          len = MAX(0.0,b_fcus->l->v)*tree->mod->ras->gamma_rr->v[i];//branch_len * rate
          len *= tree->mod->br_len_mult->v;
          if(tree->mixt_tree)  len *= tree->mixt_tree->mod->ras->gamma_rr->v[tree->mod->ras->parent_class_number];
          if(len < l_min)      len = l_min;
          else if(len > l_max) len = l_max;
          
          mean = len;
          /* var  = MAX(0.0,b_fcus->l_var->v) * POW(tree->mod->ras->gamma_rr->v[i]*tree->mod->br_len_mult->v,2); */
          /* var  = tree->mod->l_var_sigma * POW(tree->mod->ras->gamma_rr->v[i]*tree->mod->br_len_mult->v,2); */
          var  = MAX(0.0,b_fcus->l->v) * tree->mod->l_var_sigma->v * POW(tree->mod->ras->gamma_rr->v[i]*tree->mod->br_len_mult->v,2);
          if(tree->mixt_tree) var *= POW(tree->mixt_tree->mod->ras->gamma_rr->v[tree->mod->ras->parent_class_number],2);
        }

      //Update the transition prob. matrix
      if(tree->mod->gamma_mgf_bl == NO)
          {
#ifdef BEAGLE
            assert(UNINITIALIZED != tree->mod->b_inst);
#endif

            PMat(len,tree->mod,i*tree->mod->ns*tree->mod->ns,b_fcus->Pij_rr,b_fcus->tPij_rr);
          }
      else
          {
#ifdef BEAGLE
            Warn_And_Exit(TODO_BEAGLE);
#endif

            PMat_MGF_Gamma(mean,var,tree->mod,i*tree->mod->ns*tree->mod->ns,b_fcus->Pij_rr,b_fcus->tPij_rr);
          }
    }

#ifdef BEAGLE
  int whichmodel = tree->mod->whichmodel;
  //Only for some models we use Beagle to compute/update the P-matrices, for other models
  //we compute them in PhyML and explicitly set the P-matrices in BEAGLE
  if((tree->mod->io->datatype == AA || whichmodel==GTR || whichmodel==CUSTOM) && tree->mod->use_m4mod == NO)
    {
      if(b_fcus->has_zero_br_len == YES)
        Warn_And_Exit(TODO_BEAGLE);
      
      //
      update_beagle_eigen(tree->mod);
      update_beagle_ras(tree->mod);
      
      //
      len = MAX(0.0, b_fcus->l->v) * tree->mod->br_len_mult->v;
      int p_matrices[1]     = b_fcus->Pij_rr_idx;
      double branch_lens[1] = len;
      int ret = beagleUpdateTransitionMatrices(tree->b_inst,0,p_matrices,NULL,NULL,branch_lens,1);
      if(ret<0)
        {
          PhyML_Fprintf(stderr, "beagleUpdateTransitionMatrices() on instance %i failed:%i\n\n",tree->b_inst,ret);
          Exit("");
        }
      //Retrieve a "local" copy of the P-matrix
      ret = beagleGetTransitionMatrix(tree->b_inst, b_fcus->Pij_rr_idx, b_fcus->Pij_rr);
      if(ret<0)
        {
          PhyML_Fprintf(stderr, "beagleGetTransitionMatrix() on instance %i failed:%i\n\n",tree->b_inst,ret);
          Exit("");
        }
    }
  else
    {
      int ret = beagleSetTransitionMatrix(tree->b_inst, b_fcus->Pij_rr_idx, b_fcus->Pij_rr, -1);
      if(ret<0)
        {
          PhyML_Fprintf(stderr, "beagleSetTransitionMatrix() on instance %i failed:%i\n\n",tree->b_inst,ret);
          Exit("");
        }
  }
#endif
    if(b_fcus->packed_tPij_rr != NULL)
      {
        memcpy(b_fcus->packed_tPij_rr,
               b_fcus->tPij_rr,
               (size_t)tree->mod->ras->n_catg *
               (size_t)tree->mod->ns *
               (size_t)tree->mod->ns *
               sizeof(phydbl));
      }
    if(tree->mod->log_l == YES) b_fcus->l->v = log(b_fcus->l->v);

//      Print_Model(tree->mod);
//      Dump_Arr_D(tree->cur_site_lk, tree->n_pattern);
//      Print_Edge_PMats(tree, b_fcus);
//      Print_Edge_Likelihoods(tree,b_fcus,true);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Update_Partial_Lk_Along_A_Path(t_node **path, int path_length, t_tree *tree)
{
  int i,j;
  
  for(i=0;i<path_length-1;++i)
    {
      for(j=0;j<3;++j)
        if(path[i]->v[j] == path[i+1])
          {
            if(path[i] == path[i]->b[j]->left)
              {
                Update_Partial_Lk(tree,path[i]->b[j],path[i]->b[j]->left);
              }
            else if(path[i] == path[i]->b[j]->rght)
              {
                Update_Partial_Lk(tree,path[i]->b[j],path[i]->b[j]->rght);
              }
            else
              {
                PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d. \n",__FILE__,__LINE__);
                assert(FALSE);
              }
            break;
          }
#ifdef DEBUG
      if(j == 3)
        {
          PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d.\n",__FILE__,__LINE__);
          assert(FALSE);
        }
#endif
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Lk_Dist(phydbl *F, phydbl dist, t_mod *mod)
{
  int i,j,k;
  phydbl lnL,len;
  int dim1,dim2;
  phydbl pi, pijk;

  // Compute likelihood of the model made of the
  // first class of the mixture.
  /* if(mod->is_mixt_mod == YES) mod = mod->next; */
  /* assert(mod); */

  if(mod->log_l == YES) dist = exp(dist);

  for(k=0;k<mod->ras->n_catg;k++)
    {
      len = dist*mod->ras->gamma_rr->v[k];
      if(len < mod->l_min)      len = mod->l_min;
      else if(len > mod->l_max) len = mod->l_max;
      PMat(len,mod,mod->ns*mod->ns*k,mod->Pij_rr->v,NULL);
      /* PhyML_Printf("\n. p: %g len: %g",mod->Pij_rr->v[0],len); */
    }

  dim1 = mod->ns*mod->ns;
  dim2 = mod->ns;
  lnL  = .0;
  pi   = -1.;
  pijk = -1.;

  for(i=0;i<mod->ns-1;i++)
    {
      pi = mod->e_frq->pi->v[i];

      for(j=i+1;j<mod->ns;j++)
        {
          for(k=0;k<mod->ras->n_catg;k++)
            {
              pijk = mod->Pij_rr->v[dim1*k+dim2*i+j];              
              lnL += (F[dim1*k+dim2*i+j] + F[dim1*k+dim2*j+i]) * log(pi * pijk);
              /* PhyML_Printf("\n. pijk: %g lnL: %g",pijk,lnL); */
            }
        }
    }
  
  for(i=0;i<mod->ns;i++) 
    {
      pi = mod->e_frq->pi->v[i];
      
      for(k=0;k<mod->ras->n_catg;k++) 
        {
          pijk = mod->Pij_rr->v[dim1*k+dim2*i+i];
          lnL += F[dim1*k+dim2*i+i]* log(pi * pijk);
          /* PhyML_Printf("\n. pijk: %g lnL: %g",pijk,lnL); */
        }
    }

  return lnL;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Update_Lk_At_Given_Edge(t_edge *b_fcus, t_tree *tree)
{
  Update_Partial_Lk(tree,b_fcus,b_fcus->left);
  Update_Partial_Lk(tree,b_fcus,b_fcus->rght);
  tree->c_lnL = Lk(b_fcus,tree);
  return tree->c_lnL;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Init_Partial_Lk_Loc(t_tree *tree)
{
  int i,j;
  t_node *d;
  int *patt_id_d;

  if(tree->is_mixt_tree == YES) return;
  
  for(i=0;i<2*tree->n_otu-1;++i)
    {
      for(j=0;j<tree->n_pattern;j++)
        {
          tree->a_edges[i]->p_lk_loc_left[j] = j;
          tree->a_edges[i]->p_lk_loc_rght[j] = j;
        }
    }
  
  for(i=0;i<tree->n_otu;i++)
    {
      d = tree->a_nodes[i];
      patt_id_d = (d == d->b[0]->left)?(d->b[0]->patt_id_left):(d->b[0]->patt_id_rght);
      for(j=0;j<tree->n_pattern;j++)
        {
          assert(tree->a_nodes[d->num]->c_seq);
          patt_id_d[j] = (int)tree->a_nodes[d->num]->c_seq->state[j];
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Lk_Normal_Approx(t_tree *tree)
{
  phydbl lnL;
  int i;
  int dim;
  phydbl first_order;

  dim = 2*tree->n_otu-3;

  lnL = Dnorm_Multi_Given_InvCov_Det(tree->rates->u_cur_l,
                     tree->rates->mean_l,
                     tree->rates->invcov,
                     tree->rates->covdet,
                     2*tree->n_otu-3,YES);

  first_order = 0.0;
  for(i=0;i<dim;i++) first_order += (tree->rates->u_cur_l[i] - tree->rates->mean_l[i]) * tree->rates->grad_l[i];

  lnL += first_order;

  return(lnL);

}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


phydbl Wrap_Part_Lk_At_Given_Edge(t_edge *b, t_tree *tree, supert_tree *stree)
{
  return -1.0;
  /* return PART_Lk_At_Given_Edge(b,stree);; */
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Wrap_Part_Lk(t_edge *b, t_tree *tree, supert_tree *stree)
{
  return -1.0;
  /* return PART_Lk(stree); */
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


phydbl Wrap_Lk(t_edge *b, t_tree *tree, supert_tree *stree)
{
  Lk(NULL,tree);
  return tree->c_lnL;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#if (defined GEO)
phydbl Wrap_Geo_Lk(t_edge *b, t_tree *tree, supert_tree *stree)
{
  TIPO_Lk(tree);
  return tree->geo_lnL;
}
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


phydbl Wrap_Lk_At_Given_Edge(t_edge *b, t_tree *tree, supert_tree *stree)
{
  Lk(b,tree);
  return tree->c_lnL;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#if(defined PHYREX || PHYTIME)
phydbl Wrap_Lk_Rates(t_edge *b, t_tree *tree, supert_tree *stree)
{
  RATES_Lk(tree);
  return tree->rates->c_lnL;
}
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#if(defined PHYREX || PHYTIME)
phydbl Wrap_Lk_Times(t_edge *b, t_tree *tree, supert_tree *stree)
{
  TIMES_Lk(tree);
  return tree->times->c_lnL;
}
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl Wrap_Lk_Linreg(t_edge *b, t_tree *tree, supert_tree *stree)
{
  /* RATES_Lk_Linreg(tree); */
  return -1.;
  /* return tree->rates->c_lnL_linreg; */
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


phydbl Wrap_Diff_Lk_Norm_At_Given_Edge(t_edge *b, t_tree *tree, supert_tree *stree)
{
  phydbl diff;
  diff = Diff_Lk_Norm_At_Given_Edge(b,tree);
  return(-diff);

}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

int Check_Lk_At_Given_Edge(int verbose, t_tree *tree)
{
  int res;
  int i;
  phydbl *lk;

  lk = (phydbl *)mCalloc(2*tree->n_otu-3,sizeof(phydbl));

  res = 0;
  for(i=0;i<2*tree->n_otu-3;++i)
    {
      lk[i] = Lk(tree->a_edges[i],tree);
      if(verbose == YES) PhyML_Printf("\n. Edge %3d %13G %f %13G",
                                      tree->a_edges[i]->num,tree->a_edges[i]->l->v,lk[i],
                                      tree->a_edges[i]->l_var->v);
            
    }

  if(tree->n_root && tree->ignore_root == NO)
    {
      Lk(tree->n_root->b[1],tree);
      if(verbose == YES) PhyML_Printf("\nx Edge %3d %13G %f %13G",
                                      tree->n_root->b[1]->num,tree->n_root->b[1]->l->v,tree->c_lnL,
                                      tree->n_root->b[1]->l_var->v
                                      );

      Lk(tree->n_root->b[2],tree);
      if(verbose == YES) PhyML_Printf("\nx Edge %3d %13G %f %13G",
                                      tree->n_root->b[2]->num,tree->n_root->b[2]->l->v,tree->c_lnL,
                                      tree->n_root->b[2]->l_var->v
                                      );

    }

  res=1;
  for(i=1;i<2*tree->n_otu-3;i++)
    {
      if(FABS(lk[i]-lk[i-1]) > 1.E-2) res=0;
    }
  Free(lk);

  return res;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
// Computes the value of fact_sum_scale, the part of the scaling factors
// that is common to all classes of the mixture and scale the 
// likelihood for each mixture using the part of the scaling
// factors that is class-specific
 
void Pull_Scaling_Factors(int site, t_edge *b, t_tree *tree)
{
  unsigned int catg;
  const unsigned int ncatg = tree->mod->ras->n_catg;

  if(tree->apply_lk_scaling == NO)
    {
      tree->fact_sum_scale[site] = 0;
      for(catg=0;catg<ncatg;++catg) tree->unscaled_site_lk_cat[site*ncatg+catg] = tree->site_lk_cat[catg];
      return;
    }
  else
    {
      switch(tree->scaling_method)
        {
        case SCALE_RATE_SPECIFIC : 
          {
            int *sum_scale_left_cat,*sum_scale_rght_cat;
            int exponent;
            phydbl max_sum_scale,min_sum_scale;
            phydbl sum,tmp,dum;
            
            sum_scale_left_cat = b->sum_scale_left_cat;
            sum_scale_rght_cat = b->sum_scale_rght_cat;
            
            max_sum_scale =   (phydbl)BIG;
            min_sum_scale =  -(phydbl)BIG;
            
            for(catg=0;catg<ncatg;++catg)
              {
                sum_scale_left_cat[catg] =
                  (b->sum_scale_left)?
                  (b->sum_scale_left[site*ncatg+catg]):
                  (0.0);
                
                sum_scale_rght_cat[catg] =
                  (b->sum_scale_rght)?
                  (b->sum_scale_rght[site*ncatg+catg]):
                  (0.0);
                
                sum = sum_scale_left_cat[catg] + sum_scale_rght_cat[catg];
                
                if(sum < .0)
                  {
                    PhyML_Fprintf(stderr,"\n. tree: %s\n",Write_Tree(tree));
                    PhyML_Fprintf(stderr,"\n. b->num = %d  sum = %G root ? %d",sum,b->num,b == tree->e_root);
                    PhyML_Fprintf(stderr,"\n. Err. in file %s at line %d.\n",__FILE__,__LINE__);
                    Exit("\n");
                  }
                
                dum = log(FABS(tree->site_lk_cat[catg]));
                
                tmp = sum + ((phydbl)LOGBIG - dum)/(phydbl)LOG2;
                if(tmp < max_sum_scale) max_sum_scale = tmp; /* min of the maxs */
                
                tmp = sum + ((phydbl)LOGSMALL - dum)/(phydbl)LOG2;
                if(tmp > min_sum_scale) min_sum_scale = tmp; /* max of the mins */
                
                assert(isnan(tmp) == NO);
              }
            
            if(min_sum_scale > max_sum_scale)
              {
#ifdef SAFEMODE
                PhyML_Printf("\n. Numerical precision issue alert.");
                PhyML_Printf("\n. min_sum_scale = %G max_sum_scale = %G",min_sum_scale,max_sum_scale);
#endif
                min_sum_scale = max_sum_scale;
              }
            
            tree->fact_sum_scale[site] = (int)((max_sum_scale + min_sum_scale) / 2);
                        
            /* Apply scaling factors */
            for(catg=0;catg<ncatg;++catg)
              {
                exponent = -(sum_scale_left_cat[catg]+sum_scale_rght_cat[catg])+tree->fact_sum_scale[site];
                Rate_Correction(exponent,tree->site_lk_cat + catg);
              }
            
            break;
          }
        case SCALE_FAST :
          {
            int sum_scale_left,sum_scale_rght;
            
            sum_scale_left =
              (b->sum_scale_left)?
              (b->sum_scale_left[site]):
              (0.0);
            
            sum_scale_rght =
              (b->sum_scale_rght)?
              (b->sum_scale_rght[site]):
              (0.0);
            
            tree->fact_sum_scale[site] = sum_scale_left + sum_scale_rght;

            break;
          }
        default :
          {
            assert(FALSE);
            break;
          }
        }
      for(catg=0;catg<ncatg;++catg) tree->unscaled_site_lk_cat[site*ncatg+catg] = tree->site_lk_cat[catg]; 
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Tree should be ready for likelihood analysis when calling
// this function.
 void Stepwise_Add_Lk(t_tree *tree)
 {
   t_edge **residuals,**targets,*best_target;
  int *nd_idx,i,j,n_targets,*tg_idx,n_opt;

  residuals   = (t_edge **)mCalloc(tree->n_otu-3,sizeof(t_edge *));
  targets     = (t_edge **)mCalloc(2*tree->n_otu-3,sizeof(t_edge *));
  best_target = NULL;
  nd_idx      = Permutate(tree->n_otu-3);

  // Remove all tips except that corresponding to a_nodes[0], 
  // a_nodes[1] and a_nodes[2].  
  for(i=0;i<tree->n_otu-3;i++)
    {
      Prune_Subtree(tree->a_nodes[i+3]->v[0],                   
                    tree->a_nodes[i+3],
                    NULL,
                    residuals+i,
                    tree);
    }

  // Initial targets
  n_targets = 3;
  for(i=0;i<n_targets;i++) targets[i] = tree->a_nodes[i]->b[0];

  // Regraft each tip on the tree at most parsimonious position
  for(i=0;i<tree->n_otu-3;i++)
    {
      Set_Both_Sides(YES,tree);
      Lk(NULL,tree);

      printf("\n. [%d/%d]",i,tree->n_otu-3);

      tree->best_lnL = UNLIKELY;
      best_target    = NULL;
      tg_idx         = Permutate(n_targets);

      for(j=0;j<n_targets;j++)
        {
          Graft_Subtree(targets[tg_idx[j]],
                        tree->a_nodes[nd_idx[i]+3]->v[0],
                        NULL,
                        residuals[i],
                        NULL,
                        tree);
          
          Update_PMat_At_Given_Edge(targets[tg_idx[j]],tree);
          Update_PMat_At_Given_Edge(tree->a_nodes[nd_idx[i]+3]->b[0],tree);
          Update_Partial_Lk(tree,residuals[i],tree->a_nodes[nd_idx[i]+3]->v[0]);
          Lk(residuals[i],tree);

          if(tree->c_lnL > tree->best_lnL)
            {
              tree->best_lnL = tree->c_lnL;
              best_target = targets[tg_idx[j]];
            }
          
          Prune_Subtree(tree->a_nodes[nd_idx[i]+3]->v[0],                        
                        tree->a_nodes[nd_idx[i]+3],
                        NULL,
                        residuals+i,
                        tree);
        }

      assert(best_target);
            
      Graft_Subtree(best_target,
                    tree->a_nodes[nd_idx[i]+3]->v[0],
                    NULL,
                    residuals[i],
                    NULL,
                    tree);
      
      n_opt = 0;
      do Optimize_Br_Len_Serie (2,tree); while(n_opt++ < 3);

      targets[n_targets]   = residuals[i]; 
      targets[n_targets+1] = tree->a_nodes[nd_idx[i]+3]->b[0];
      
      Free(tg_idx);
      n_targets+=2;
    }

  Round_Optimize(tree,5);
  PhyML_Fprintf(stderr,"\n. lk: %f",tree->c_lnL);
  Exit("\n");

  Free(nd_idx);
  Free(residuals);
  Free(targets);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

/*     |
       |
       |b
       |
       |d
      /  \
     /    \
    /      \
   /        \
  /v1        \v2

  Set p_lk and sum_scale for subtrees with d, v1 and v2 as root,
  Pij for edges b, and the two edges connecting d to v1 and d to
  v2;
  Account for rooted trees.
*/

void Set_All_Partial_Lk(t_node **n_v1, t_node **n_v2,
                        phydbl **p_lk, int **sum_scale, int **p_lk_loc,
                        phydbl **Pij1, phydbl **tPij1, phydbl **p_lk_v1, int **sum_scale_v1,
                        phydbl **Pij2, phydbl **tPij2, phydbl **p_lk_v2, int **sum_scale_v2,
                        t_node *d, t_edge *b, t_tree *tree
#ifdef BEAGLE
                        , int *dest_p_idx, int *child1_p_idx, int* child2_p_idx, int* Pij1_idx, int* Pij2_idx
#endif
                        )
{
  unsigned int i;
  
  assert(tree->is_mixt_tree == NO);
  assert(d->tax == NO);
  
  if(tree->n_root == NULL || tree->ignore_root == YES)
    {
      /* Does d lie on the "left" or "right" of the branch? */
      if(d == b->left)
        {
          *p_lk      = b->p_lk_left;
          *sum_scale = b->sum_scale_left;
#ifdef BEAGLE
          *dest_p_idx = b->p_lk_left_idx;
#endif
        }
      else
        {
          *p_lk      = b->p_lk_rght;
          *sum_scale = b->sum_scale_rght;
#ifdef BEAGLE
          *dest_p_idx = b->p_lk_rght_idx;
#endif
        }

      *n_v1 = *n_v2 = NULL;
      for(i=0;i<3;++i)
        {
          if(d->b[i] != b)
            {
              if(!(*n_v1))
                {
                  *n_v1 = d->v[i];
#ifdef BEAGLE
                  Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,d->b[i],tree,child1_p_idx,Pij1_idx);
#else
                  Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,d->b[i],tree);
#endif
                }
              else if(!(*n_v2))
                {
                  *n_v2 = d->v[i];
#ifdef BEAGLE
                  Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree,child2_p_idx,Pij2_idx);
#else
                  Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree);
#endif
                }
              else
                {
                  PhyML_Printf("\n. Issue detected with node %d.\n",d->num);
                  assert(FALSE);
                }
            }
        }
    }
  else
    {
      if(b == tree->e_root)
        {
          if(d == tree->n_root->v[1])      b = tree->n_root->b[1];
          else if(d == tree->n_root->v[2]) b = tree->n_root->b[2];
          else assert(FALSE);
        }

      if(d == tree->n_root)
        {
          if(b == tree->n_root->b[1])
            {
              *p_lk      = tree->n_root->b[1]->p_lk_left;
              *sum_scale = tree->n_root->b[1]->sum_scale_left;
#ifdef BEAGLE
              *dest_p_idx = tree->n_root->b[1]->p_lk_left_idx;
#endif
            }
          else
            {
              *p_lk      = tree->n_root->b[2]->p_lk_left;
              *sum_scale = tree->n_root->b[2]->sum_scale_left;
#ifdef BEAGLE
              *dest_p_idx = tree->n_root->b[2]->p_lk_left_idx;
#endif
            }

          *n_v1         = NULL;
          *Pij1         = NULL;
          *tPij1        = NULL;
          *p_lk_v1      = NULL;
          *sum_scale_v1 = NULL;

          if(b == tree->n_root->b[1])
            {
              *n_v2         = tree->n_root->v[2];
              *Pij2         = tree->n_root->b[2]->Pij_rr;
              *tPij2        = tree->n_root->b[2]->tPij_rr;
              *p_lk_v2      = tree->n_root->b[2]->p_lk_rght;
              *sum_scale_v2 = tree->n_root->b[2]->sum_scale_rght;
#ifdef BEAGLE
              *child2_p_idx = tree->n_root->b[2]->p_lk_rght_idx;
              *Pij2_idx     = tree->n_root->b[2]->Pij_rr_idx;
#endif
            }
          else if(b == tree->n_root->b[2])
            {
              *n_v2         = tree->n_root->v[1];
              *Pij2         = tree->n_root->b[1]->Pij_rr;
              *tPij2        = tree->n_root->b[1]->tPij_rr;
              *p_lk_v2      = tree->n_root->b[1]->p_lk_rght;
              *sum_scale_v2 = tree->n_root->b[1]->sum_scale_rght;
#ifdef BEAGLE
              *child2_p_idx = tree->n_root->b[1]->p_lk_rght_idx;
              *Pij2_idx     = tree->n_root->b[1]->Pij_rr_idx;
#endif
            }
          else assert(FALSE);
        }
      else if(d == tree->n_root->v[1] || d == tree->n_root->v[2])
        {
          if(b == tree->n_root->b[1] || b == tree->n_root->b[2])
            {
              if(b == tree->n_root->b[1])
                {
                  *p_lk      = tree->n_root->b[1]->p_lk_rght;
                  *sum_scale = tree->n_root->b[1]->sum_scale_rght;
#ifdef BEAGLE
                  *dest_p_idx = tree->n_root->b[1]->p_lk_rght_idx;
#endif
                }
              else
                {
                  *p_lk      = tree->n_root->b[2]->p_lk_rght;
                  *sum_scale = tree->n_root->b[2]->sum_scale_rght;
#ifdef BEAGLE
                  *dest_p_idx = tree->n_root->b[2]->p_lk_rght_idx;
#endif
                }

              *n_v1 = *n_v2 = NULL;
              for(i=0;i<3;++i)
                {
                  if(d->b[i] != tree->e_root)
                    {
                      if(!(*n_v1))
                        {
                          *n_v1 = d->v[i];
#ifdef BEAGLE
                          Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,d->b[i],tree,child1_p_idx,Pij1_idx);
#else
                          Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,d->b[i],tree);
#endif
                        }
                      else
                        {
                          *n_v2 = d->v[i];
#ifdef BEAGLE
                          Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree,child2_p_idx,Pij2_idx);
#else
                          Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree);
#endif
                        }
                    }
                }
            }
          else
            {
              if(d == b->left)
                {
                  *p_lk      = b->p_lk_left;
                  *sum_scale = b->sum_scale_left;
                  *p_lk_loc  = b->p_lk_loc_left;
#ifdef BEAGLE
                  *dest_p_idx = b->p_lk_left_idx;
#endif
                }
              else
                {
                  *p_lk      = b->p_lk_rght;
                  *sum_scale = b->sum_scale_rght;
                  *p_lk_loc  = b->p_lk_loc_rght;
#ifdef BEAGLE
                  *dest_p_idx = b->p_lk_rght_idx;
#endif
                }


              *n_v1 = tree->n_root;
#ifdef BEAGLE
              Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,
                                      (d == tree->n_root->v[1])?
                                      (tree->n_root->b[1]):
                                      (tree->n_root->b[2]),
                                      tree,child1_p_idx,Pij1_idx);
#else
              Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,
                                      (d == tree->n_root->v[1])?
                                      (tree->n_root->b[1]):
                                      (tree->n_root->b[2]),
                                      tree);
#endif
              for(i=0;i<3;i++)
                {
                  if(d->b[i] != tree->e_root && d->b[i] != b)
                    {
                      *n_v2 = d->v[i];
#ifdef BEAGLE
                      Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree,child2_p_idx,Pij2_idx);
#else
                      Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree);
#endif
                      break;
                    }
                }

            }
        }
      else
        {
          if(d == b->left)
            {
              *p_lk      = b->p_lk_left;
              *sum_scale = b->sum_scale_left;
              *p_lk_loc  = b->p_lk_loc_left;
#ifdef BEAGLE
              *dest_p_idx = b->p_lk_left_idx;
#endif
            }
          else
            {
              *p_lk      = b->p_lk_rght;
              *sum_scale = b->sum_scale_rght;
              *p_lk_loc  = b->p_lk_loc_rght;
#ifdef BEAGLE
              *dest_p_idx = b->p_lk_rght_idx;
#endif
            }

          *n_v1 = *n_v2 = NULL;
          for(i=0;i<3;i++)
            {
              if(d->b[i] != b)
                {
                  if(!(*n_v1))
                    {
                      *n_v1 = d->v[i];
#ifdef BEAGLE
                      Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,d->b[i],tree,child1_p_idx,Pij1_idx);
#else
                      Set_Partial_Lk_One_Side(Pij1,tPij1,p_lk_v1,sum_scale_v1,d,d->b[i],tree);
#endif
                    }
                  else
                    {
                      *n_v2 = d->v[i];
#ifdef BEAGLE
                      Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree,child2_p_idx,Pij2_idx);
#else
                      Set_Partial_Lk_One_Side(Pij2,tPij2,p_lk_v2,sum_scale_v2,d,d->b[i],tree);
#endif
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

/*     |
       |
       |d
      /  \
     /    \
    /      \b
   /        \
  /          \x (either n_v1 or n_v2)

  Returns p_lk and sum_scale for subtree with x as root, Pij for edge b
*/

void Set_Partial_Lk_One_Side(phydbl **Pij, phydbl **tPij, phydbl **p_lk,  int **sum_scale, t_node *d, t_edge *b, t_tree *tree
#ifdef BEAGLE
                                     , int* child_p_idx, int* Pij_idx
#endif
                                     )
{

  if(Pij != NULL)
    {
      *Pij  = b->Pij_rr;
      *tPij = b->tPij_rr;
#ifdef BEAGLE
      *Pij_idx = b->Pij_rr_idx;
#endif
    }

  if(d->tax == NO)
    {
      if(d == b->left) // if d is on the left of b, then d's neighbor is on the right
        {
          *p_lk      = (b->rght->tax == YES) ? b->p_lk_tip_r : b->p_lk_rght;
          *sum_scale = b->sum_scale_rght;
#ifdef BEAGLE
          *child_p_idx = b->rght->tax? b->p_lk_tip_idx: b->p_lk_rght_idx;
#endif

          if(*p_lk == NULL) PhyML_Printf("\n. b:%d b->left:%d b->rght:%d d:%d",
                                         b->num,
                                         b->left->num,
                                         b->rght->num,
                                         d->num);
          assert(*p_lk);
        }
      else
        {
          *p_lk      = b->p_lk_left;
          *sum_scale = b->sum_scale_left;
#ifdef BEAGLE
          *child_p_idx   = b->rght->tax? b->p_lk_tip_idx: b->p_lk_left_idx;
#endif

          if(*p_lk == NULL) PhyML_Printf("\n. b:%d b->left:%d b->rght:%d d:%d",
                                         b->num,
                                         b->left->num,
                                         b->rght->num,
                                         d->num);
          assert(*p_lk);
        }
    }
  else
    {
#ifdef BEAGLE
      Warn_And_Exit(TODO_BEAGLE);
#endif
      *p_lk        = NULL;
      *sum_scale   = NULL;
      PhyML_Printf("\n. WARNING. p_lk set to NULL. d->num: %d b->num: %d",d->num,b->num);
        
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Switch_Partial_Lk_Post(t_node *a, t_node *d, t_edge *b, short int yesno, t_tree *tree)
{
  if(a == tree->n_root) assert(FALSE);
  if(d->tax == NO)
    {
      int i;

      for(i=0;i<3;++i)
        if(d->v[i] != a)
          Switch_Partial_Lk_Post(d,d->v[i],d->b[i],yesno,tree);
    } 

  if(b->left == d) b->update_partial_lk_left = yesno;
  else if(b->rght == d) b->update_partial_lk_rght = yesno;
  else assert(FALSE);

  return;
}


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Switch_Partial_Lk_Pre(t_node *a, t_node *d, t_edge *b, short int yesno, t_tree *tree)
{  
  if(a == tree->n_root) assert(FALSE);
  if(d->tax == YES) return;
  else
    {
      int i;
      
      for(i=0;i<3;++i)
        {
          if(d->v[i] != a)
            {
              if(d->b[i]->left == d) d->b[i]->update_partial_lk_left = yesno;
              else if(d->b[i]->rght == d) d->b[i]->update_partial_lk_rght = yesno;
              else assert(FALSE);
            }
        }
      
      for(i=0;i<3;++i)
        if(d->v[i] != a)
          Switch_Partial_Lk_Pre(d,d->v[i],d->b[i],yesno,tree);
    }
  
  return;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
#if PHYML_OPT_PARTIAL_LK
static inline phydbl Partial_Lk_Max_4(const phydbl x0, const phydbl x1,
                                      const phydbl x2, const phydbl x3)
{
  phydbl largest_p_lk = x0;

  if(x1 > largest_p_lk) largest_p_lk = x1;
  if(x2 > largest_p_lk) largest_p_lk = x2;
  if(x3 > largest_p_lk) largest_p_lk = x3;

  return largest_p_lk;
}

static inline int Partial_Lk_All_One_4(const phydbl *plk1, const phydbl *plk2)
{
  return (plk1[0] == 1.0 && plk1[1] == 1.0 && plk1[2] == 1.0 && plk1[3] == 1.0 &&
          plk2[0] == 1.0 && plk2[1] == 1.0 && plk2[2] == 1.0 && plk2[3] == 1.0);
}

static inline phydbl Partial_Lk_Inin_4_Max(const phydbl *Pij1, const phydbl *plk1,
                                           const phydbl *Pij2, const phydbl *plk2,
                                           phydbl *plk0)
{
  if(Partial_Lk_All_One_4(plk1,plk2))
    {
      plk0[0] = plk0[1] = plk0[2] = plk0[3] = 1.0;
      return 1.0;
    }

  {
    const phydbl p10 = plk1[0], p11 = plk1[1], p12 = plk1[2], p13 = plk1[3];
    const phydbl p20 = plk2[0], p21 = plk2[1], p22 = plk2[2], p23 = plk2[3];

    const phydbl u10 = Pij1[0]*p10 + Pij1[1]*p11 + Pij1[2]*p12 + Pij1[3]*p13;
    const phydbl u11 = Pij1[4]*p10 + Pij1[5]*p11 + Pij1[6]*p12 + Pij1[7]*p13;
    const phydbl u12 = Pij1[8]*p10 + Pij1[9]*p11 + Pij1[10]*p12 + Pij1[11]*p13;
    const phydbl u13 = Pij1[12]*p10 + Pij1[13]*p11 + Pij1[14]*p12 + Pij1[15]*p13;

    const phydbl u20 = Pij2[0]*p20 + Pij2[1]*p21 + Pij2[2]*p22 + Pij2[3]*p23;
    const phydbl u21 = Pij2[4]*p20 + Pij2[5]*p21 + Pij2[6]*p22 + Pij2[7]*p23;
    const phydbl u22 = Pij2[8]*p20 + Pij2[9]*p21 + Pij2[10]*p22 + Pij2[11]*p23;
    const phydbl u23 = Pij2[12]*p20 + Pij2[13]*p21 + Pij2[14]*p22 + Pij2[15]*p23;
    const phydbl x0 = u10*u20;
    const phydbl x1 = u11*u21;
    const phydbl x2 = u12*u22;
    const phydbl x3 = u13*u23;

    plk0[0] = x0;
    plk0[1] = x1;
    plk0[2] = x2;
    plk0[3] = x3;

    return Partial_Lk_Max_4(x0,x1,x2,x3);
  }
}

static inline void Partial_Lk_Inin_4(const phydbl *Pij1, const phydbl *plk1,
                                     const phydbl *Pij2, const phydbl *plk2,
                                     phydbl *plk0)
{
  Partial_Lk_Inin_4_Max(Pij1,plk1,Pij2,plk2,plk0);
}

static inline phydbl Partial_Lk_Exex_4_Max(const phydbl *Pij1, const int state1,
                                           const phydbl *Pij2, const int state2,
                                           phydbl *plk0)
{
  const phydbl x0 = Pij1[state1]      * Pij2[state2];
  const phydbl x1 = Pij1[4 + state1]  * Pij2[4 + state2];
  const phydbl x2 = Pij1[8 + state1]  * Pij2[8 + state2];
  const phydbl x3 = Pij1[12 + state1] * Pij2[12 + state2];

  plk0[0] = x0;
  plk0[1] = x1;
  plk0[2] = x2;
  plk0[3] = x3;

  return Partial_Lk_Max_4(x0,x1,x2,x3);
}

static inline void Partial_Lk_Exex_4(const phydbl *Pij1, const int state1,
                                     const phydbl *Pij2, const int state2,
                                     phydbl *plk0)
{
  Partial_Lk_Exex_4_Max(Pij1,state1,Pij2,state2,plk0);
}

static inline phydbl Partial_Lk_Exin_4_Max(const phydbl *Pij1, const int state1,
                                           const phydbl *Pij2, const phydbl *plk2,
                                           phydbl *plk0)
{
  const phydbl p20 = plk2[0], p21 = plk2[1], p22 = plk2[2], p23 = plk2[3];

  const phydbl u20 = Pij2[0]*p20 + Pij2[1]*p21 + Pij2[2]*p22 + Pij2[3]*p23;
  const phydbl u21 = Pij2[4]*p20 + Pij2[5]*p21 + Pij2[6]*p22 + Pij2[7]*p23;
  const phydbl u22 = Pij2[8]*p20 + Pij2[9]*p21 + Pij2[10]*p22 + Pij2[11]*p23;
  const phydbl u23 = Pij2[12]*p20 + Pij2[13]*p21 + Pij2[14]*p22 + Pij2[15]*p23;
  const phydbl x0 = Pij1[state1]      * u20;
  const phydbl x1 = Pij1[4 + state1]  * u21;
  const phydbl x2 = Pij1[8 + state1]  * u22;
  const phydbl x3 = Pij1[12 + state1] * u23;

  plk0[0] = x0;
  plk0[1] = x1;
  plk0[2] = x2;
  plk0[3] = x3;

  return Partial_Lk_Max_4(x0,x1,x2,x3);
}

static inline void Partial_Lk_Exin_4(const phydbl *Pij1, const int state1,
                                     const phydbl *Pij2, const phydbl *plk2,
                                     phydbl *plk0)
{
  Partial_Lk_Exin_4_Max(Pij1,state1,Pij2,plk2,plk0);
}

#define PARTIAL_LK_ACC_DOT20(sum,row,plk) \
  do { \
    (sum) += (row)[0]  * (plk)[0];  \
    (sum) += (row)[1]  * (plk)[1];  \
    (sum) += (row)[2]  * (plk)[2];  \
    (sum) += (row)[3]  * (plk)[3];  \
    (sum) += (row)[4]  * (plk)[4];  \
    (sum) += (row)[5]  * (plk)[5];  \
    (sum) += (row)[6]  * (plk)[6];  \
    (sum) += (row)[7]  * (plk)[7];  \
    (sum) += (row)[8]  * (plk)[8];  \
    (sum) += (row)[9]  * (plk)[9];  \
    (sum) += (row)[10] * (plk)[10]; \
    (sum) += (row)[11] * (plk)[11]; \
    (sum) += (row)[12] * (plk)[12]; \
    (sum) += (row)[13] * (plk)[13]; \
    (sum) += (row)[14] * (plk)[14]; \
    (sum) += (row)[15] * (plk)[15]; \
    (sum) += (row)[16] * (plk)[16]; \
    (sum) += (row)[17] * (plk)[17]; \
    (sum) += (row)[18] * (plk)[18]; \
    (sum) += (row)[19] * (plk)[19]; \
  } while(0)

static inline int Partial_Lk_All_One_20(const phydbl *plk1, const phydbl *plk2)
{
  return (plk1[0] == 1.0 && plk1[1] == 1.0 && plk1[2] == 1.0 && plk1[3] == 1.0 &&
          plk1[4] == 1.0 && plk1[5] == 1.0 && plk1[6] == 1.0 && plk1[7] == 1.0 &&
          plk1[8] == 1.0 && plk1[9] == 1.0 && plk1[10] == 1.0 && plk1[11] == 1.0 &&
          plk1[12] == 1.0 && plk1[13] == 1.0 && plk1[14] == 1.0 && plk1[15] == 1.0 &&
          plk1[16] == 1.0 && plk1[17] == 1.0 && plk1[18] == 1.0 && plk1[19] == 1.0 &&
          plk2[0] == 1.0 && plk2[1] == 1.0 && plk2[2] == 1.0 && plk2[3] == 1.0 &&
          plk2[4] == 1.0 && plk2[5] == 1.0 && plk2[6] == 1.0 && plk2[7] == 1.0 &&
          plk2[8] == 1.0 && plk2[9] == 1.0 && plk2[10] == 1.0 && plk2[11] == 1.0 &&
          plk2[12] == 1.0 && plk2[13] == 1.0 && plk2[14] == 1.0 && plk2[15] == 1.0 &&
          plk2[16] == 1.0 && plk2[17] == 1.0 && plk2[18] == 1.0 && plk2[19] == 1.0);
}

static inline phydbl Partial_Lk_Inin_20_Max(const phydbl *Pij1, const phydbl *plk1,
                                            const phydbl *Pij2, const phydbl *plk2,
                                            phydbl *plk0)
{
  unsigned int i;
  phydbl largest_p_lk;
  const phydbl *row1 = Pij1;
  const phydbl *row2 = Pij2;

  if(Partial_Lk_All_One_20(plk1,plk2))
    {
      for(i=0;i<20;++i) plk0[i] = 1.0;
      return 1.0;
    }

  largest_p_lk = -BIG;

  for(i=0;i<20;++i)
    {
      phydbl u1 = 0.0;
      phydbl u2 = 0.0;
      phydbl x;

      PARTIAL_LK_ACC_DOT20(u1,row1,plk1);
      PARTIAL_LK_ACC_DOT20(u2,row2,plk2);

      x = u1*u2;
      plk0[i] = x;
      if(x > largest_p_lk) largest_p_lk = x;
      row1 += 20;
      row2 += 20;
    }

  return largest_p_lk;
}

static inline void Partial_Lk_Inin_20(const phydbl *Pij1, const phydbl *plk1,
                                      const phydbl *Pij2, const phydbl *plk2,
                                      phydbl *plk0)
{
  Partial_Lk_Inin_20_Max(Pij1,plk1,Pij2,plk2,plk0);
}

static inline phydbl Partial_Lk_Exex_20_Max(const phydbl *Pij1, const int state1,
                                            const phydbl *Pij2, const int state2,
                                            phydbl *plk0)
{
  unsigned int i;
  const phydbl *col1 = Pij1 + state1;
  const phydbl *col2 = Pij2 + state2;
  phydbl largest_p_lk = -BIG;

  for(i=0;i<20;++i)
    {
      const phydbl x = col1[0]*col2[0];
      plk0[i] = x;
      if(x > largest_p_lk) largest_p_lk = x;
      col1 += 20;
      col2 += 20;
    }

  return largest_p_lk;
}

static inline void Partial_Lk_Exex_20(const phydbl *Pij1, const int state1,
                                      const phydbl *Pij2, const int state2,
                                      phydbl *plk0)
{
  Partial_Lk_Exex_20_Max(Pij1,state1,Pij2,state2,plk0);
}

static inline phydbl Partial_Lk_Exin_20_Max(const phydbl *Pij1, const int state1,
                                            const phydbl *Pij2, const phydbl *plk2,
                                            phydbl *plk0)
{
  unsigned int i;
  const phydbl *col1 = Pij1 + state1;
  const phydbl *row2 = Pij2;
  phydbl largest_p_lk = -BIG;

  for(i=0;i<20;++i)
    {
      phydbl u2 = 0.0;
      phydbl x;

      PARTIAL_LK_ACC_DOT20(u2,row2,plk2);

      x = col1[0]*u2;
      plk0[i] = x;
      if(x > largest_p_lk) largest_p_lk = x;
      col1 += 20;
      row2 += 20;
    }

  return largest_p_lk;
}

static inline void Partial_Lk_Exin_20(const phydbl *Pij1, const int state1,
                                      const phydbl *Pij2, const phydbl *plk2,
                                      phydbl *plk0)
{
  Partial_Lk_Exin_20_Max(Pij1,state1,Pij2,plk2,plk0);
}

#undef PARTIAL_LK_ACC_DOT20
#endif

void Partial_Lk_Inin(const phydbl *Pij1, const phydbl *plk1, const phydbl *Pij2, const phydbl *plk2, const int ns, phydbl *plk0)
{
  unsigned int i,j;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      Partial_Lk_Inin_4(Pij1,plk1,Pij2,plk2,plk0);
      return;
    }
  if(ns == 20)
    {
      Partial_Lk_Inin_20(Pij1,plk1,Pij2,plk2,plk0);
      return;
    }
#endif

  
  for(i=0;i<ns;++i) if(plk1[i] > 1.0 || plk1[i] < 1.0 || plk2[i] > 1.0 || plk2[i] < 1.0) break; 

  if(i != ns)
    {
      for(i=0;i<ns;++i)
        {
          phydbl u1 = 0.0;
          phydbl u2 = 0.0;

          for(j=0;j<ns;++j)
            {
              u1 += Pij1[j] * plk1[j];
              u2 += Pij2[j] * plk2[j];
            }
          
          Pij1 += ns;
          Pij2 += ns;
          plk0[i] = u1*u2;
        }
    }
  else
    {
      for(i=0;i<ns;++i) plk0[i] = 1.0;
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Partial_Lk_Exex(const phydbl *Pij1, const int state1, const phydbl *Pij2, const int state2, const int ns, phydbl *plk0)
{
  unsigned int i;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      Partial_Lk_Exex_4(Pij1,state1,Pij2,state2,plk0);
      return;
    }
  if(ns == 20)
    {
      Partial_Lk_Exex_20(Pij1,state1,Pij2,state2,plk0);
      return;
    }
#endif

  for(i=0;i<ns;++i)
    {
      plk0[i] = Pij1[state1]*Pij2[state2];
      Pij1 += ns;
      Pij2 += ns;      
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void Partial_Lk_Exin(const phydbl *Pij1, const int state1, const phydbl *Pij2, const phydbl *plk2, const int ns, phydbl *plk0)
{
  unsigned int i,j;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      Partial_Lk_Exin_4(Pij1,state1,Pij2,plk2,plk0);
      return;
    }
  if(ns == 20)
    {
      Partial_Lk_Exin_20(Pij1,state1,Pij2,plk2,plk0);
      return;
    }
#endif
  
  for(i=0;i<ns;++i)
    {
      phydbl u2 = 0.0;
      for(j=0;j<ns;++j) u2 += Pij2[j] * plk2[j];
      plk0[i] = Pij1[state1]*u2;
      Pij1 += ns;
      Pij2 += ns;
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


/* log(Variance) = log_var_multiplier + log(dt) */
phydbl Sample_Ancestral_Trait_Contmod(t_node *a, t_node *d, phydbl t_za, phydbl t_zd, phydbl log_var_multiplier, int print, t_tree *tree)
{
  /*     
         a
        /\ 
       /  \
      z    \
     /
    /
    d
   /\
  /  \

  */
  
  phydbl mu_up,mu_down;
  phydbl var_zd,var_za,var_up,var_down;
  phydbl mean,sd;
  
  mu_up   = tree->contmod->mu_up[d->num];
  mu_down = tree->contmod->mu_down[d->num];
  
  var_up   = tree->contmod->var_up[d->num];
  var_down = tree->contmod->var_down[d->num];

  var_zd = log_var_multiplier;

  var_za = var_zd;
  var_za += log(t_za);
  var_za = exp(var_za);

  if(t_zd > SMALL)
    {    
      var_zd += log(t_zd);    
      var_zd = exp(var_zd);
    }
  else
    {
      var_zd = 0.0;
    }

  if(!(var_zd+var_za+var_up+var_down > 0.0))
    {
      PhyML_Printf("\n. a: %d d: %d t_za: %f t_zd: %f var_up: %f var_down = %f var_zd = %f var_za = %f",
                   a->num,d->num,
                   t_za,t_zd,
                   var_up,var_down,
                   var_zd,var_za);
    }
  assert(var_zd+var_za+var_up+var_down > 0.0);
  
  mean = (mu_down*(var_za+var_up) + mu_up*(var_zd+var_down))/(var_zd+var_za+var_up+var_down);
  sd = sqrt((var_za+var_up)*(var_zd+var_down)/(var_za+var_zd+var_up+var_down));
  
  if(print == YES)
    {
      PhyML_Printf("\n. log_var_multiplier: %f mu_up: %f mu_down: %f var_up: %f var_down: %f mean: %f sd: %f",
                   log_var_multiplier,
                   mu_up,mu_down,
                   var_up,var_down,
                   mean,sd);
    }

  assert(isnan(sd) == NO && isinf(sd) == NO);
  assert(isnan(mean) == NO && isinf(mean) == NO);
  
  return(Rnorm(mean,sd));
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
