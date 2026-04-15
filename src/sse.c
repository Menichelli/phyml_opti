/*

PhyML:  a program that  computes maximum likelihood phylogenies from
DNA or AA homologous sequences.

Copyright (C) Stephane Guindon. Oct 2003 onward.

All parts of the source except where indicated are distributed under
the GNU public licence. See http://www.opensource.org for details.

*/

#include "assert.h"
#include "sse.h"
#ifdef __ARM_NEON
#include "sse2neon.h"
#endif

phydbl Lk_Site_Eigen_Local(unsigned int site,
                           const phydbl *expl, const phydbl *dot_prod,
                           t_edge *b, t_tree *tree,
                           phydbl *site_lk_cat_local, int *site_warning);

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#if (((defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__ARM_NEON)) && !((defined __AVX__ || defined __AVX2__))) && !defined(DISABLE_NATIVE))

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

static inline phydbl SSE_Vect_Max(__m128d x);
static inline phydbl SSE_Vects_Max(const __m128d *x, unsigned int nblocks);

#if PHYML_MT_LK_RUNTIME
static int PhyML_MT_Force_Max_Threads_SSE(void)
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

static int PhyML_MT_Recommended_SSE_Threads(long long work, long long min_work_per_thread)
{
  const int max_threads = omp_get_max_threads();
  int threads;

  if(max_threads <= 1) return 1;
  if(PhyML_MT_Force_Max_Threads_SSE() == 1) return max_threads;
  if(max_threads <= 1 || work < min_work_per_thread) return 1;

  threads = (int)((work + min_work_per_thread - 1LL) / min_work_per_thread);
  if(threads < 1) threads = 1;
  if(threads > max_threads) threads = max_threads;
  return threads;
}

static int PhyML_MT_Threads_SSE_Update_Partial_Lk(int npatterns, int ncatg, int ns)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns;
  return PhyML_MT_Recommended_SSE_Threads(work,224000LL);
}

static int PhyML_MT_Threads_SSE_Update_Eigen_Lr(int npatterns, int ncatg, int ns)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns;
  return PhyML_MT_Recommended_SSE_Threads(work,128000LL);
}

static void PhyML_MT_SSE_Get_Site_Range(unsigned int nsites, unsigned int *begin, unsigned int *end)
{
  const unsigned int tid = (unsigned int)omp_get_thread_num();
  const unsigned int nth = (unsigned int)omp_get_num_threads();

  *begin = (unsigned int)(((unsigned long long)nsites * tid) / nth);
  *end   = (unsigned int)(((unsigned long long)nsites * (tid + 1U)) / nth);
}

static void SSE_Prepare_Eigen_Packs(t_tree *tree)
{
  unsigned int i,j;
  const unsigned int ns = tree->mod->ns;
  const unsigned int sz = (unsigned int)BYTE_ALIGN / 8U;
  const unsigned int nblocks = ns / sz;
  phydbl *l_ev,*r_ev;

  if(tree->eigen_pack_valid == YES &&
     tree->eigen_pack_epoch == tree->mod->eigen_epoch)
    return;

  l_ev = tree->l_ev;
  r_ev = tree->mod->eigen->r_e_vect;

  for(i=0;i<ns;++i)
    for(j=0;j<ns;++j)
      l_ev[i*ns+j] = tree->mod->eigen->l_e_vect[j*ns+i];

  l_ev = tree->l_ev;
  for(i=0;i<ns;++i)
    {
      for(j=0;j<nblocks;++j)
        {
          tree->_r_ev[i*nblocks+j] = _mm_load_pd(r_ev + j*sz);
          tree->_l_ev[i*nblocks+j] = _mm_load_pd(l_ev + j*sz);
        }
      r_ev += ns;
      l_ev += ns;
    }

  tree->eigen_pack_epoch = tree->mod->eigen_epoch;
  tree->eigen_pack_valid = YES;
}

static const __m128d *SSE_Find_Packed_tPij(const t_tree *tree, const phydbl *raw_tPij)
{
  int i;

  if(raw_tPij == NULL) return NULL;

  for(i=0;i<2*tree->n_otu-1;++i)
    {
      const t_edge *edge = tree->a_edges[i];

      if(edge != NULL && edge->tPij_rr == raw_tPij)
        {
          assert(edge->packed_tPij_rr != NULL);
          return (const __m128d *)edge->packed_tPij_rr;
        }
    }

  assert(FALSE);
  return NULL;
}
#endif

#if PHYML_OPT_PARTIAL_LK
static inline int SSE_All_One(const phydbl *plk, unsigned int ns)
{
  unsigned int i;
  for(i=0;i<ns;++i) if(plk[i] != 1.0) return 0;
  return 1;
}

static inline void SSE_Matrix_Vect_Prod_4(const __m128d *_m_transpose, const phydbl *_v, __m128d *_u)
{
  const __m128d x0 = _mm_set1_pd(_v[0]);
  const __m128d x1 = _mm_set1_pd(_v[1]);
  const __m128d x2 = _mm_set1_pd(_v[2]);
  const __m128d x3 = _mm_set1_pd(_v[3]);

  _u[0] = _mm_add_pd(_mm_add_pd(_mm_mul_pd(_m_transpose[0],x0),
                                _mm_mul_pd(_m_transpose[2],x1)),
                     _mm_add_pd(_mm_mul_pd(_m_transpose[4],x2),
                                _mm_mul_pd(_m_transpose[6],x3)));
  _u[1] = _mm_add_pd(_mm_add_pd(_mm_mul_pd(_m_transpose[1],x0),
                                _mm_mul_pd(_m_transpose[3],x1)),
                     _mm_add_pd(_mm_mul_pd(_m_transpose[5],x2),
                                _mm_mul_pd(_m_transpose[7],x3)));
}

static inline void SSE_Matrix_Vect_Prod_20(const __m128d *_m_transpose, const phydbl *_v, __m128d *_u)
{
  unsigned int i;
  __m128d x;

  x = _mm_set1_pd(_v[0]);
  for(i=0;i<10;++i) _u[i] = _mm_mul_pd(_m_transpose[i],x);

  for(i=1;i<20;++i)
    {
      unsigned int j;
      const __m128d *row = _m_transpose + 10*i;
      x = _mm_set1_pd(_v[i]);
      for(j=0;j<10;++j) _u[j] = _mm_add_pd(_u[j],_mm_mul_pd(row[j],x));
    }
}

static inline void SSE_Partial_Lk_Exex_4(const __m128d *_tPij1, const int state1,
                                         const __m128d *_tPij2, const int state2,
                                         __m128d *plk0)
{
  const __m128d *col1 = _tPij1 + 2*state1;
  const __m128d *col2 = _tPij2 + 2*state2;

  plk0[0] = _mm_mul_pd(col1[0],col2[0]);
  plk0[1] = _mm_mul_pd(col1[1],col2[1]);
}

static inline void SSE_Partial_Lk_Exex_20(const __m128d *_tPij1, const int state1,
                                          const __m128d *_tPij2, const int state2,
                                          __m128d *plk0)
{
  unsigned int i;
  const __m128d *col1 = _tPij1 + 10*state1;
  const __m128d *col2 = _tPij2 + 10*state2;

  for(i=0;i<10;++i) plk0[i] = _mm_mul_pd(col1[i],col2[i]);
}

static inline phydbl SSE_Partial_Lk_Exex_4_Max(const __m128d *_tPij1, const int state1,
                                               const __m128d *_tPij2, const int state2,
                                               __m128d *plk0)
{
  SSE_Partial_Lk_Exex_4(_tPij1,state1,_tPij2,state2,plk0);
  return MAX(SSE_Vect_Max(plk0[0]),SSE_Vect_Max(plk0[1]));
}

static inline phydbl SSE_Partial_Lk_Exex_20_Max(const __m128d *_tPij1, const int state1,
                                                const __m128d *_tPij2, const int state2,
                                                __m128d *plk0)
{
  unsigned int i;
  phydbl largest_p_lk = -BIG;

  SSE_Partial_Lk_Exex_20(_tPij1,state1,_tPij2,state2,plk0);
  for(i=0;i<10;++i)
    if(SSE_Vect_Max(plk0[i]) > largest_p_lk)
      largest_p_lk = SSE_Vect_Max(plk0[i]);

  return largest_p_lk;
}

static inline void SSE_Partial_Lk_Exin_4(const __m128d *_tPij1, const int state1,
                                         const __m128d *_tPij2, const phydbl *_plk2,
                                         __m128d *_pmat2plk2, __m128d *_plk0)
{
  const __m128d *col1 = _tPij1 + 2*state1;

  SSE_Matrix_Vect_Prod_4(_tPij2,_plk2,_pmat2plk2);
  _plk0[0] = _mm_mul_pd(col1[0],_pmat2plk2[0]);
  _plk0[1] = _mm_mul_pd(col1[1],_pmat2plk2[1]);
}

static inline void SSE_Partial_Lk_Exin_20(const __m128d *_tPij1, const int state1,
                                          const __m128d *_tPij2, const phydbl *_plk2,
                                          __m128d *_pmat2plk2, __m128d *_plk0)
{
  unsigned int i;
  const __m128d *col1 = _tPij1 + 10*state1;

  SSE_Matrix_Vect_Prod_20(_tPij2,_plk2,_pmat2plk2);
  for(i=0;i<10;++i) _plk0[i] = _mm_mul_pd(col1[i],_pmat2plk2[i]);
}

static inline phydbl SSE_Partial_Lk_Exin_4_Max(const __m128d *_tPij1, const int state1,
                                               const __m128d *_tPij2, const phydbl *_plk2,
                                               __m128d *_pmat2plk2, __m128d *_plk0)
{
  SSE_Partial_Lk_Exin_4(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
  return MAX(SSE_Vect_Max(_plk0[0]),SSE_Vect_Max(_plk0[1]));
}

static inline phydbl SSE_Partial_Lk_Exin_20_Max(const __m128d *_tPij1, const int state1,
                                                const __m128d *_tPij2, const phydbl *_plk2,
                                                __m128d *_pmat2plk2, __m128d *_plk0)
{
  unsigned int i;
  phydbl largest_p_lk = -BIG;

  SSE_Partial_Lk_Exin_20(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
  for(i=0;i<10;++i)
    if(SSE_Vect_Max(_plk0[i]) > largest_p_lk)
      largest_p_lk = SSE_Vect_Max(_plk0[i]);

  return largest_p_lk;
}

static inline void SSE_Partial_Lk_Inin_4(const __m128d *_tPij1, const phydbl *plk1,
                                         __m128d *_pmat1plk1, const __m128d *_tPij2,
                                         const phydbl *plk2, __m128d *_pmat2plk2,
                                         __m128d *_plk0)
{
  if(SSE_All_One(plk1,4) && SSE_All_One(plk2,4))
    {
      _plk0[0] = _mm_set1_pd(1.0);
      _plk0[1] = _mm_set1_pd(1.0);
      return;
    }

  SSE_Matrix_Vect_Prod_4(_tPij1,plk1,_pmat1plk1);
  SSE_Matrix_Vect_Prod_4(_tPij2,plk2,_pmat2plk2);
  _plk0[0] = _mm_mul_pd(_pmat1plk1[0],_pmat2plk2[0]);
  _plk0[1] = _mm_mul_pd(_pmat1plk1[1],_pmat2plk2[1]);
}

static void SSE_Partial_Lk_Inin_20(const __m128d *_tPij1, const phydbl *plk1,
                                   __m128d *_pmat1plk1, const __m128d *_tPij2,
                                   const phydbl *plk2, __m128d *_pmat2plk2,
                                   __m128d *_plk0)
{
  unsigned int i;
  __m128d u2[10];

  (void)_pmat1plk1;
  (void)_pmat2plk2;

  if(SSE_All_One(plk1,20) && SSE_All_One(plk2,20))
    {
      for(i=0;i<10;++i) _plk0[i] = _mm_set1_pd(1.0);
      return;
    }

  SSE_Matrix_Vect_Prod_20(_tPij1,plk1,_plk0);
  SSE_Matrix_Vect_Prod_20(_tPij2,plk2,u2);
  for(i=0;i<10;++i) _plk0[i] = _mm_mul_pd(_plk0[i],u2[i]);
}

static inline phydbl SSE_Partial_Lk_Inin_4_Max(const __m128d *_tPij1, const phydbl *plk1,
                                               __m128d *_pmat1plk1, const __m128d *_tPij2,
                                               const phydbl *plk2, __m128d *_pmat2plk2,
                                               __m128d *_plk0)
{
  SSE_Partial_Lk_Inin_4(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
  return MAX(SSE_Vect_Max(_plk0[0]),SSE_Vect_Max(_plk0[1]));
}

static inline phydbl SSE_Partial_Lk_Inin_20_Max(const __m128d *_tPij1, const phydbl *plk1,
                                                __m128d *_pmat1plk1, const __m128d *_tPij2,
                                                const phydbl *plk2, __m128d *_pmat2plk2,
                                                __m128d *_plk0)
{
  unsigned int i;
  phydbl largest_p_lk = -BIG;

  SSE_Partial_Lk_Inin_20(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
  for(i=0;i<10;++i)
    if(SSE_Vect_Max(_plk0[i]) > largest_p_lk)
      largest_p_lk = SSE_Vect_Max(_plk0[i]);

  return largest_p_lk;
}

static inline phydbl SSE_Partial_Lk_Exex_Max(const __m128d *_tPij1, const int state1,
                                             const __m128d *_tPij2, const int state2,
                                             const int ns, __m128d *plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  phydbl largest_p_lk = -BIG;
  unsigned int i;

  if(ns == 4) return SSE_Partial_Lk_Exex_4_Max(_tPij1,state1,_tPij2,state2,plk0);
  if(ns == 20) return SSE_Partial_Lk_Exex_20_Max(_tPij1,state1,_tPij2,state2,plk0);

  _tPij1 = _tPij1 + state1 * nblocks;
  _tPij2 = _tPij2 + state2 * nblocks;
  for(i=0;i<nblocks;++i)
    {
      const __m128d x = _mm_mul_pd(_tPij1[i],_tPij2[i]);
      const phydbl block_max = SSE_Vect_Max(x);
      plk0[i] = x;
      if(block_max > largest_p_lk) largest_p_lk = block_max;
    }

  return largest_p_lk;
}

static inline phydbl SSE_Partial_Lk_Exin_Max(const __m128d *_tPij1, const int state1,
                                             const __m128d *_tPij2, const phydbl *_plk2,
                                             __m128d *_pmat2plk2, const int ns, __m128d *_plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  phydbl largest_p_lk = -BIG;
  unsigned int i;

  if(ns == 4) return SSE_Partial_Lk_Exin_4_Max(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
  if(ns == 20) return SSE_Partial_Lk_Exin_20_Max(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);

  _tPij1 = _tPij1 + state1 * nblocks;
  SSE_Matrix_Vect_Prod(_tPij2,_plk2,ns,_pmat2plk2);
  for(i=0;i<nblocks;++i)
    {
      const __m128d x = _mm_mul_pd(_tPij1[i],_pmat2plk2[i]);
      const phydbl block_max = SSE_Vect_Max(x);
      _plk0[i] = x;
      if(block_max > largest_p_lk) largest_p_lk = block_max;
    }

  return largest_p_lk;
}

static inline phydbl SSE_Partial_Lk_Inin_Max(const __m128d *_tPij1, const phydbl *plk1,
                                             __m128d *_pmat1plk1, const __m128d *_tPij2,
                                             const phydbl *plk2, __m128d *_pmat2plk2,
                                             const int ns, __m128d *_plk0)
{
  unsigned int i;
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  phydbl largest_p_lk = -BIG;

  if(ns == 4) return SSE_Partial_Lk_Inin_4_Max(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
  if(ns == 20) return SSE_Partial_Lk_Inin_20_Max(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);

  for(i=0;i<ns;++i) if(plk1[i] > 1.0 || plk1[i] < 1.0 || plk2[i] > 1.0 || plk2[i] < 1.0) break;

  if(i != ns)
    {
      SSE_Matrix_Vect_Prod(_tPij1,plk1,ns,_pmat1plk1);
      SSE_Matrix_Vect_Prod(_tPij2,plk2,ns,_pmat2plk2);
      for(i=0;i<nblocks;++i)
        {
          const __m128d x = _mm_mul_pd(_pmat1plk1[i],_pmat2plk2[i]);
          const phydbl block_max = SSE_Vect_Max(x);
          _plk0[i] = x;
          if(block_max > largest_p_lk) largest_p_lk = block_max;
        }
    }
  else
    {
      for(i=0;i<nblocks;++i) _plk0[i] = _mm_set1_pd(1.0);
      largest_p_lk = 1.0;
    }

  return largest_p_lk;
}

#if PHYML_MT_LK_RUNTIME
static void SSE_Copy_tPij_Local(const phydbl *src_tPij1, const phydbl *src_tPij2,
                                __m128d *dst_tPij1, __m128d *dst_tPij2,
                                const unsigned int ns, const unsigned int ncatg)
{
  const unsigned int sz = (unsigned int)BYTE_ALIGN / 8U;
  const unsigned int nblocks = ns / sz;
  unsigned int i,j,k;

  for(i=0;i<ncatg;++i)
    {
      for(j=0;j<ns;++j)
        {
          for(k=0;k<nblocks;++k)
            {
              dst_tPij1[k] = _mm_load_pd(src_tPij1);
              dst_tPij2[k] = _mm_load_pd(src_tPij2);
              src_tPij1 += sz;
              src_tPij2 += sz;
            }
          dst_tPij1 += nblocks;
          dst_tPij2 += nblocks;
        }
    }
}

static void SSE_Update_Partial_Lk_Prepared_Range(t_tree *tree,
                                                 const t_node *n_v1, const t_node *n_v2,
                                                 phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                 int *sum_scale, const int *sum_scale_v1, const int *sum_scale_v2,
                                                 const __m128d *init_tPij1, const __m128d *init_tPij2,
                                                 const unsigned int site_begin, const unsigned int site_end,
                                                 const unsigned int ns, const unsigned int ncatg,
                                                 t_lk_thread_ctx *ctx)
{
  const unsigned int ncatgns = ncatg * ns;
  const unsigned int nsns = ns * ns;
  const unsigned int sz = (unsigned int)BYTE_ALIGN / 8U;
  const unsigned int nblocks = ns / sz;
  const unsigned int tmat_catg_step = nsns / sz;
  const int tax_v1 = (n_v1->tax != 0);
  const int tax_v2 = (n_v2->tax != 0);
  const int scale_fast = (tree->scaling_method == SCALE_FAST);
  const int do_scaling = (scale_fast && tree->apply_lk_scaling == YES);
  const int plk1_catg_step = (tax_v1) ? 0 : (int)ns;
  const int plk2_catg_step = (tax_v2) ? 0 : (int)ns;
  const int plk1_site_stride = (tax_v1) ? (int)ns : (int)ncatgns;
  const int plk2_site_stride = (tax_v2) ? (int)ns : (int)ncatgns;
  const short int *is_ambigu_v1 = (tax_v1) ? n_v1->c_seq->is_ambigu : NULL;
  const short int *is_ambigu_v2 = (tax_v2) ? n_v2->c_seq->is_ambigu : NULL;
  const short int *d_state_v1 = (tax_v1) ? n_v1->c_seq->d_state : NULL;
  const short int *d_state_v2 = (tax_v2) ? n_v2->c_seq->d_state : NULL;
  __m128d *pmat1plk1_local;
  __m128d *pmat2plk2_local;
  __m128d *plk0_local;
  unsigned int site;

  assert(ctx != NULL);

  pmat1plk1_local = ctx->_pmat1plk1;
  pmat2plk2_local = ctx->_pmat2plk2;
  plk0_local      = ctx->_plk0;

  for(site=site_begin;site<site_end;++site)
    {
      unsigned int catg,k;
      short int state_v1 = -1;
      short int state_v2 = -1;
      short int ambiguity_check_v1 = YES;
      short int ambiguity_check_v2 = YES;
      phydbl largest_p_lk = -BIG;
      phydbl catg_largest_p_lk;
      phydbl *site_plk0;
      const phydbl *site_plk1,*site_plk2;
      const __m128d *site_tPij1,*site_tPij2;

      if(tree->data->wght[site] <= SMALL) continue;

      site_plk0 = plk0 + (size_t)site * ncatgns;
      site_plk1 = plk1 + (size_t)site * plk1_site_stride;
      site_plk2 = plk2 + (size_t)site * plk2_site_stride;
      site_tPij1 = init_tPij1;
      site_tPij2 = init_tPij2;

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

      for(catg=0;catg<ncatg;++catg)
        {
          phydbl *catg_plk0 = site_plk0 + catg * ns;
          const phydbl *catg_plk1 = site_plk1 + catg * plk1_catg_step;
          const phydbl *catg_plk2 = site_plk2 + catg * plk2_catg_step;

          if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
            {
              if(do_scaling)
                {
                  catg_largest_p_lk = SSE_Partial_Lk_Exex_Max(site_tPij1,state_v1,
                                                              site_tPij2,state_v2,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  SSE_Partial_Lk_Exex(site_tPij1,state_v1,site_tPij2,state_v2,ns,plk0_local);
                }
            }
          else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
            {
              if(do_scaling)
                {
                  catg_largest_p_lk = SSE_Partial_Lk_Exin_Max(site_tPij2,state_v2,
                                                              site_tPij1,catg_plk1,pmat1plk1_local,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  SSE_Partial_Lk_Exin(site_tPij2,state_v2,
                                      site_tPij1,catg_plk1,pmat1plk1_local,
                                      ns,plk0_local);
                }
            }
          else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
            {
              if(do_scaling)
                {
                  catg_largest_p_lk = SSE_Partial_Lk_Exin_Max(site_tPij1,state_v1,
                                                              site_tPij2,catg_plk2,pmat2plk2_local,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  SSE_Partial_Lk_Exin(site_tPij1,state_v1,
                                      site_tPij2,catg_plk2,pmat2plk2_local,
                                      ns,plk0_local);
                }
            }
          else
            {
              if(do_scaling)
                {
                  catg_largest_p_lk = SSE_Partial_Lk_Inin_Max(site_tPij1,catg_plk1,pmat1plk1_local,
                                                              site_tPij2,catg_plk2,pmat2plk2_local,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  SSE_Partial_Lk_Inin(site_tPij1,catg_plk1,pmat1plk1_local,
                                      site_tPij2,catg_plk2,pmat2plk2_local,
                                      ns,plk0_local);
                }
            }

          for(k=0;k<nblocks;++k) _mm_store_pd(catg_plk0 + sz*k,plk0_local[k]);
          site_tPij1 += tmat_catg_step;
          site_tPij2 += tmat_catg_step;
        }

      if(scale_fast)
        {
          sum_scale[site] = (sum_scale_v1 ? sum_scale_v1[site] : 0) +
                            (sum_scale_v2 ? sum_scale_v2[site] : 0);

          if(do_scaling && largest_p_lk < INV_TWO_TO_THE_LARGE &&
             tree->mod->augmented == NO &&
             tree->apply_lk_scaling == YES)
            {
              for(k=0;k<ncatgns;++k) site_plk0[k] *= TWO_TO_THE_LARGE;
              sum_scale[site] += LARGE;
            }
        }
    }
}

static void SSE_Update_Partial_Lk_Prepared_Team(t_tree *tree,
                                                const t_node *n_v1, const t_node *n_v2,
                                                phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                int *sum_scale, const int *sum_scale_v1, const int *sum_scale_v2,
                                                const __m128d *init_tPij1, const __m128d *init_tPij2,
                                                const unsigned int npattern, const unsigned int ns, const unsigned int ncatg,
                                                t_lk_thread_ctx *ctx)
{
  unsigned int begin,end;

  assert(ctx != NULL);

  PhyML_MT_SSE_Get_Site_Range(npattern,&begin,&end);
  SSE_Update_Partial_Lk_Prepared_Range(tree,
                                       n_v1,n_v2,
                                       plk0,plk1,plk2,
                                       sum_scale,sum_scale_v1,sum_scale_v2,
                                       init_tPij1,init_tPij2,
                                       begin,end,ns,ncatg,ctx);
}

static void SSE_Update_Partial_Lk_MT(t_tree *tree,
                                     const t_node *n_v1, const t_node *n_v2,
                                     phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                     int *sum_scale, const int *sum_scale_v1, const int *sum_scale_v2,
                                     const __m128d *init_tPij1, const __m128d *init_tPij2,
                                     const unsigned int npattern, const unsigned int ns, const unsigned int ncatg,
                                     const int nthreads)
{
  #pragma omp parallel num_threads(nthreads)
    {
      t_lk_thread_ctx *ctx = tree->lk_thread_ctx + omp_get_thread_num();
      SSE_Update_Partial_Lk_Prepared_Team(tree,
                                          n_v1,n_v2,
                                          plk0,plk1,plk2,
                                          sum_scale,sum_scale_v1,sum_scale_v2,
                                          init_tPij1,init_tPij2,
                                          npattern,ns,ncatg,ctx);
    }
}
#endif
#endif

#if PHYML_MT_LK_RUNTIME
#if PHYML_OPT_PARTIAL_LK
void SSE_Update_Partial_Lk_Team(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx)
{
  t_node *n_v1, *n_v2;
  phydbl *plk0,*plk1,*plk2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;
  const unsigned int npattern = tree->n_pattern;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const __m128d *init_tPij1,*init_tPij2;

  assert(ctx != NULL);

  n_v1 = n_v2                 = NULL;
  plk0 = plk1 = plk2          = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  sum_scale = sum_scale_v1 = sum_scale_v2 = NULL;
  p_lk_loc                    = NULL;

  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &plk0,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&plk1,&sum_scale_v1,
                     &Pij2,&tPij2,&plk2,&sum_scale_v2,
                     d,b,tree);

  if(tree->mod->augmented == YES)
    {
      #pragma omp single
      {
        PhyML_Printf("\n== SSE version of the Update_Partial_Lk function does not");
        PhyML_Printf("\n== allow augmented data.");
        assert(FALSE);
      }
      return;
    }

  init_tPij1 = SSE_Find_Packed_tPij(tree,tPij1);
  init_tPij2 = SSE_Find_Packed_tPij(tree,tPij2);

  SSE_Update_Partial_Lk_Prepared_Team(tree,
                                      n_v1,n_v2,
                                      plk0,plk1,plk2,
                                      sum_scale,sum_scale_v1,sum_scale_v2,
                                      init_tPij1,init_tPij2,
                                      npattern,ns,ncatg,ctx);
}

void SSE_Update_Partial_Lk_Wavefront_Job(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx)
{
  t_node *n_v1, *n_v2;
  phydbl *plk0,*plk1,*plk2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;
  const unsigned int npattern = tree->n_pattern;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;

  assert(ctx != NULL);

  n_v1 = n_v2                 = NULL;
  plk0 = plk1 = plk2          = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  sum_scale = sum_scale_v1 = sum_scale_v2 = NULL;
  p_lk_loc                    = NULL;

  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &plk0,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&plk1,&sum_scale_v1,
                     &Pij2,&tPij2,&plk2,&sum_scale_v2,
                     d,b,tree);

  if(tree->mod->augmented == YES)
    {
      PhyML_Printf("\n== SSE version of the Update_Partial_Lk function does not");
      PhyML_Printf("\n== allow augmented data.");
      assert(FALSE);
    }

  SSE_Update_Partial_Lk_Prepared_Range(tree,
                                       n_v1,n_v2,
                                       plk0,plk1,plk2,
                                       sum_scale,sum_scale_v1,sum_scale_v2,
                                       SSE_Find_Packed_tPij(tree,tPij1),
                                       SSE_Find_Packed_tPij(tree,tPij2),
                                       0U,npattern,ns,ncatg,ctx);
}
#endif

static void SSE_Update_Eigen_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx, int fused_site_lk)
{
  unsigned int site,catg;
  unsigned int i;

  const unsigned int npattern = tree->n_pattern;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const unsigned int ns = tree->mod->ns;
  const unsigned int sz = (int)BYTE_ALIGN / 8;
  const unsigned int nblocks = ns / sz;
  const unsigned int ncatgns = ncatg*ns;

  const phydbl *pi;
  __m128d *_l_ev,*_r_ev;
  phydbl *p_lk_left_pi;
  __m128d *prod_left,*prod_rght;

  assert(ctx != NULL);
  assert(sz == 2);
  assert(tree->update_eigen_lr == YES);

  p_lk_left_pi = ctx->p_lk_left_pi;
  prod_left    = ctx->_prod_left;
  prod_rght    = ctx->_prod_rght;
  _l_ev        = tree->_l_ev;
  _r_ev        = tree->_r_ev;
  pi           = tree->mod->e_frq->pi->v;

  #pragma omp single
  SSE_Prepare_Eigen_Packs(tree);

  #pragma omp for schedule(static)
  for(site=0;site<npattern;++site)
    {
      phydbl *site_dot_prod = fused_site_lk ? ctx->site_dot_prod : (tree->dot_prod + (size_t)site * ncatgns);
      const phydbl *site_p_lk_left = b->left->tax ?
        (b->p_lk_tip_l + (size_t)site * ns) :
        (b->p_lk_left + (size_t)site * ncatgns);
      const phydbl *site_p_lk_rght = b->rght->tax ?
        (b->p_lk_tip_r + (size_t)site * ns) :
        (b->p_lk_rght + (size_t)site * ncatgns);

      if(tree->data->wght[site] > SMALL)
        {
          int site_warning = NO;

          for(catg=0;catg<ncatg;++catg)
            {
              for(i=0;i<ns;++i) p_lk_left_pi[i] = site_p_lk_left[i] * pi[i];

              SSE_Matrix_Vect_Prod(_r_ev,p_lk_left_pi,ns,prod_left);
              SSE_Matrix_Vect_Prod(_l_ev,site_p_lk_rght,ns,prod_rght);

              for(i=0;i<nblocks;++i)
                _mm_store_pd(site_dot_prod + i*sz,_mm_mul_pd(prod_left[i],prod_rght[i]));

              site_dot_prod += ns;
              if(b->left->tax == NO) site_p_lk_left += ns;
              if(b->rght->tax == NO) site_p_lk_rght += ns;
            }

          if(fused_site_lk)
            {
              memcpy(tree->dot_prod + (size_t)site * ncatgns,
                     ctx->site_dot_prod,
                     (size_t)ncatgns * sizeof(phydbl));
              Lk_Site_Eigen_Local(site,
                                  expl,
                                  ctx->site_dot_prod,
                                  b,tree,
                                  ctx->site_lk_cat,
                                  &site_warning);
              if(site_warning == YES) ctx->numerical_warning = YES;
            }
        }
    }
}

void SSE_Update_Eigen_Lr_Team(t_edge *b, t_tree *tree, t_lk_thread_ctx *ctx)
{
  SSE_Update_Eigen_Sites_Team(b,tree,NULL,ctx,NO);
}

void SSE_Update_Eigen_And_Lk_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx)
{
  SSE_Update_Eigen_Sites_Team(b,tree,expl,ctx,YES);
}
#endif

void SSE_Update_Eigen_Lr(t_edge *b, t_tree *tree)
{
  unsigned int site,catg;
  unsigned int i;
  
  unsigned const int npattern = tree->n_pattern;
  unsigned const int ncatg = tree->mod->ras->n_catg;
  unsigned const int ns = tree->mod->ns;
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  unsigned const int ncatgns = ncatg*ns;

  const phydbl *p_lk_left,*p_lk_rght,*pi;
  phydbl *dot_prod,*p_lk_left_pi;

  __m128d *_l_ev,*_r_ev,*_prod_left,*_prod_rght;
  
  p_lk_left_pi = tree->p_lk_left_pi;
  _l_ev        = tree->_l_ev;
  _r_ev        = tree->_r_ev;
  _prod_left   = tree->_prod_left;
  _prod_rght   = tree->_prod_rght;
  
  assert(sz == 2);
  assert(tree->update_eigen_lr == YES);

  SSE_Prepare_Eigen_Packs(tree);

  p_lk_left = b->left->tax ? b->p_lk_tip_l : b->p_lk_left;
  p_lk_rght = b->rght->tax ? b->p_lk_tip_r : b->p_lk_rght;
  pi = tree->mod->e_frq->pi->v;
  dot_prod = tree->dot_prod;

#if PHYML_MT_LK_RUNTIME
  {
    const int nthreads = PhyML_MT_Threads_SSE_Update_Eigen_Lr((int)npattern,(int)ncatg,(int)ns);

    if(nthreads > 1 && tree->lk_thread_ctx != NULL)
      {
        #pragma omp parallel num_threads(nthreads)
        {
          t_lk_thread_ctx *ctx = tree->lk_thread_ctx + omp_get_thread_num();
          SSE_Update_Eigen_Lr_Team(b,tree,ctx);
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
              for(i=0;i<ns;++i) p_lk_left_pi[i] = p_lk_left[i] * pi[i];
              
              SSE_Matrix_Vect_Prod(_r_ev,p_lk_left_pi,ns,_prod_left);
              SSE_Matrix_Vect_Prod(_l_ev,p_lk_rght,ns,_prod_rght);
              
              for(i=0;i<nblocks;++i) _mm_store_pd(dot_prod + i*sz,_mm_mul_pd(_prod_left[i],_prod_rght[i]));
              
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
          else p_lk_left += ncatgns;

          if(b->rght->tax == YES) p_lk_rght += ns;          
          else p_lk_rght += ncatgns;          

          dot_prod += ncatgns;          
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl SSE_Lk_Core_One_Class_Eigen_Lr(phydbl *dot_prod, phydbl *expl, int ns)
{
  phydbl lk;
  unsigned int l;
  const unsigned sz = (int)BYTE_ALIGN / 8;
  const unsigned int nblocks = ns/sz;
  __m128d _prod[nblocks],_x;
    
  for(l=0;l<nblocks;++l) _prod[l] = _mm_load_pd(dot_prod + l*sz);
  if(expl != NULL) for(l=0;l<nblocks;++l) _prod[l] = _mm_mul_pd(_prod[l],_mm_load_pd(expl + l*sz));
  _x = _mm_setzero_pd();
  for(l=0;l<nblocks;++l) _x = _mm_add_pd(_x,_prod[l]);

#if(defined(__SSE3__))
  _x = _mm_hadd_pd(_x,_x);
#else
  PhyML_Printf("\n. SSE3 required. Try turning on the '-msse3' or '-march=native' or '-mcpu=native' option in configure.ac.");
  assert(false);
#endif
  
  _mm_store_sd(&lk,_x);
  
  return lk;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void SSE_Lk_dLk_Core_One_Class_Eigen_Lr(phydbl *dot_prod, phydbl *expl, unsigned int ns, phydbl *lk, phydbl *dlk)
{
  unsigned int i;
  __m128d _x,_y,_z;

  _z = _mm_setzero_pd();

  for(i=0;i<ns;++i)
    {
      _x = _mm_set1_pd(dot_prod[i]);      
      _y = _mm_load_pd(expl + 2*i);

      _z = _mm_add_pd(_z,_mm_mul_pd(_x,_y));
    }
  
  *lk = ((double *)&_z)[0];
  *dlk = ((double *)&_z)[1];
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl SSE_Lk_Core_One_Class_No_Eigen_Lr(phydbl *p_lk_left, phydbl *p_lk_rght, phydbl *Pij, phydbl *tPij, phydbl *pi, int ns, int ambiguity_check, int state)
{
  phydbl lk,dum;
  const unsigned int sz = (int)BYTE_ALIGN / 8;
  const unsigned nblocks = ns/sz;
  unsigned int i,j;
  __m128d _plk;
  __m128d _plk_l[nblocks],_plk_r[nblocks];
  
  /* [ Pi . Lkr ]' x Pij x Lkl */
  
  if(ambiguity_check == NO) // tip case.
    {
      for(i=0;i<nblocks;++i)
        {
          _plk_l[i] = _mm_load_pd(p_lk_left);      
          _plk_r[i] = _mm_load_pd(Pij + state*ns);
          p_lk_left += sz;
          Pij += sz;
        }
      
      for(i=0;i<nblocks;++i)
        {
          _plk_r[i] = _mm_mul_pd(_plk_r[i],_mm_set1_pd(pi[state]));
          _plk_r[i] = _mm_mul_pd(_plk_r[i],_plk_l[i]);
        }
      
      _plk = _mm_setzero_pd();
      lk = 0.0;
      for(i=0;i<nblocks;++i)
        {

#if(defined(__SSE3__) || defined(__SSE2__) || defined(__SSE__))
          _plk = _mm_hadd_pd(_plk_r[i],_plk_r[i]);
#else
          PhyML_Printf("\n. SSE3 required. Try turning on the '-msse3' or '-march=native' or '-mcpu=native' option in configure.ac.");
          assert(false);
#endif
          
          _mm_store_sd(&dum,_plk);
          lk += dum;
        }
      return lk;
    }
  else
    {
      __m128d _pij[nblocks],_pijplk[nblocks]; 

      for(i=0;i<nblocks;++i)
        {
          _plk_r[i] = _mm_mul_pd(_mm_load_pd(p_lk_rght),_mm_load_pd(pi));
          p_lk_rght += sz;
          pi += sz;
        }

      for(i=0;i<nblocks;++i) _pijplk[i] = _mm_setzero_pd();

      for(i=0;i<ns;++i)
        {
          for(j=0;j<nblocks;++j)
            {
              _pij[j] = _mm_load_pd(tPij);
              tPij += sz;
              
              _pijplk[j] = _mm_add_pd(_pijplk[j],_mm_mul_pd(_pij[j],_mm_set1_pd(p_lk_left[i])));
            }
        }

      lk = 0.0;
      for(i=0;i<nblocks;++i)
        {
          _plk = _mm_mul_pd(_pijplk[i],_plk_r[i]);

#if(defined(__SSE3__) || defined(__SSE2__) || defined(__SSE__))
          _plk = _mm_hadd_pd(_plk,_plk);
#else
          PhyML_Printf("\n. SSE3 required. Try turning on the '-msse3' or '-march=native' or '-mcpu=native' option in configure.ac.");
          assert(false);
#endif
          
          _mm_store_sd(&dum,_plk);
          lk += dum;
        }
      return lk;
    }
  return UNLIKELY;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void SSE_Update_Partial_Lk(t_tree *tree, t_edge *b, t_node *d)
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
  phydbl *plk0,*plk1,*plk2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int sum_scale_v1_val, sum_scale_v2_val;
  unsigned int i,k;
  unsigned int catg,site;
  short int state_v1,state_v2;
  short int ambiguity_check_v1,ambiguity_check_v2;
  phydbl largest_p_lk = -BIG;
  phydbl catg_largest_p_lk = -BIG;
  int *p_lk_loc;
  
  const unsigned int npattern = tree->n_pattern;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;

  const unsigned int ncatgns =  ncatg * ns;
  const unsigned int nsns =  ns * ns;
  
  const unsigned int sz = (int)BYTE_ALIGN / 8;
  const unsigned nblocks = ns/sz;
  const unsigned int tmat_catg_step = nsns / sz;

  __m128d *_tPij1,*_tPij2,*_pmat1plk1,*_pmat2plk2,*_plk0;
  const __m128d *init_tPij1,*init_tPij2;
  int tax_v1, tax_v2;
  const int scale_fast = (tree->scaling_method == SCALE_FAST);
  const int do_scaling = (scale_fast && tree->apply_lk_scaling == YES);
  int plk1_catg_step, plk2_catg_step;
  int plk1_site_step, plk2_site_step;
  int plk1_zero_wght_step, plk2_zero_wght_step;
  const short int *is_ambigu_v1;
  const short int *is_ambigu_v2;
  const short int *d_state_v1;
  const short int *d_state_v2;
  
  _tPij1     = tree->_tPij1;
  _tPij2     = tree->_tPij2;
  _pmat1plk1 = tree->_pmat1plk1;
  _pmat2plk2 = tree->_pmat2plk2;
  _plk0      = tree->_plk0;

  sum_scale_v1_val            = 0;
  sum_scale_v2_val            = 0;
  n_v1 = n_v2                 = NULL;
  plk0 = plk1 = plk2          = NULL;
  Pij1 = Pij2                 = NULL;
  tPij1 = tPij2               = NULL;
  sum_scale_v1 = sum_scale_v2 = NULL;
  p_lk_loc                    = NULL;
  state_v1 = state_v2         = -1;

  
  Set_All_Partial_Lk(&n_v1,&n_v2,
                     &plk0,&sum_scale,&p_lk_loc,
                     &Pij1,&tPij1,&plk1,&sum_scale_v1,
                     &Pij2,&tPij2,&plk2,&sum_scale_v2,
                     d,b,tree);

  tax_v1 = (n_v1->tax != 0);
  tax_v2 = (n_v2->tax != 0);
  plk1_catg_step = (tax_v1) ? 0 : ns;
  plk2_catg_step = (tax_v2) ? 0 : ns;
  plk1_site_step = (tax_v1) ? ns : 0;
  plk2_site_step = (tax_v2) ? ns : 0;
  plk1_zero_wght_step = (tax_v1) ? ns : ncatgns;
  plk2_zero_wght_step = (tax_v2) ? ns : ncatgns;
  is_ambigu_v1 = (tax_v1) ? n_v1->c_seq->is_ambigu : NULL;
  is_ambigu_v2 = (tax_v2) ? n_v2->c_seq->is_ambigu : NULL;
  d_state_v1   = (tax_v1) ? n_v1->c_seq->d_state : NULL;
  d_state_v2   = (tax_v2) ? n_v2->c_seq->d_state : NULL;

  init_tPij1 = SSE_Find_Packed_tPij(tree,tPij1);
  init_tPij2 = SSE_Find_Packed_tPij(tree,tPij2);

  if(tree->mod->augmented == YES)
    {
      PhyML_Printf("\n== AVX version of the Update_Partial_Lk function does not");
      PhyML_Printf("\n== allow augmented data.");
      assert(FALSE);
    }

#if PHYML_MT_LK_RUNTIME && PHYML_OPT_PARTIAL_LK
  {
    const int nthreads = PhyML_MT_Threads_SSE_Update_Partial_Lk((int)npattern,(int)ncatg,(int)ns);

    if(nthreads > 1)
    {
      SSE_Update_Partial_Lk_MT(tree,
                               n_v1,n_v2,
                               plk0,plk1,plk2,
                               sum_scale,sum_scale_v1,sum_scale_v2,
                               init_tPij1,init_tPij2,
                               npattern,ns,ncatg,nthreads);
      return;
    }
  }
#endif
    
  /* For every site in the alignment */
  for(site=0;site<npattern;++site)
    {
      if(tree->data->wght[site] > SMALL)
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
          
          #if PHYML_OPT_PARTIAL_LK
            _tPij1 = (__m128d *)init_tPij1;
            _tPij2 = (__m128d *)init_tPij2;
            if(do_scaling) largest_p_lk = -BIG;

            if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
              {
                for(catg=0;catg<ncatg;++catg)
                  {
                    if(do_scaling)
                      {
                        catg_largest_p_lk = SSE_Partial_Lk_Exex_Max(_tPij1,state_v1,
                                                                    _tPij2,state_v2,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        SSE_Partial_Lk_Exex(_tPij1,state_v1,
                                            _tPij2,state_v2,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm_store_pd(plk0+sz*k,_plk0[k]);
                    _tPij1 += tmat_catg_step;
                    _tPij2 += tmat_catg_step;
                    plk0 += ns;
                    plk1 += plk1_catg_step;
                    plk2 += plk2_catg_step;
                  }
              }
            else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
              {
                for(catg=0;catg<ncatg;++catg)
                  {
                    if(do_scaling)
                      {
                        catg_largest_p_lk = SSE_Partial_Lk_Exin_Max(_tPij2,state_v2,
                                                                    _tPij1,plk1,_pmat1plk1,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        SSE_Partial_Lk_Exin(_tPij2,state_v2,
                                            _tPij1,plk1,_pmat1plk1,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm_store_pd(plk0+sz*k,_plk0[k]);
                    _tPij1 += tmat_catg_step;
                    _tPij2 += tmat_catg_step;
                    plk0 += ns;
                    plk1 += plk1_catg_step;
                    plk2 += plk2_catg_step;
                  }
              }
            else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
              {
                for(catg=0;catg<ncatg;++catg)
                  {
                    if(do_scaling)
                      {
                        catg_largest_p_lk = SSE_Partial_Lk_Exin_Max(_tPij1,state_v1,
                                                                    _tPij2,plk2,_pmat2plk2,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        SSE_Partial_Lk_Exin(_tPij1,state_v1,
                                            _tPij2,plk2,_pmat2plk2,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm_store_pd(plk0+sz*k,_plk0[k]);
                    _tPij1 += tmat_catg_step;
                    _tPij2 += tmat_catg_step;
                    plk0 += ns;
                    plk1 += plk1_catg_step;
                    plk2 += plk2_catg_step;
                  }
              }
            else
              {
                for(catg=0;catg<ncatg;++catg)
                  {
                    if(do_scaling)
                      {
                        catg_largest_p_lk = SSE_Partial_Lk_Inin_Max(_tPij1,plk1,_pmat1plk1,
                                                                    _tPij2,plk2,_pmat2plk2,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        SSE_Partial_Lk_Inin(_tPij1,plk1,_pmat1plk1,
                                            _tPij2,plk2,_pmat2plk2,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm_store_pd(plk0+sz*k,_plk0[k]);
                    _tPij1 += tmat_catg_step;
                    _tPij2 += tmat_catg_step;
                    plk0 += ns;
                    plk1 += plk1_catg_step;
                    plk2 += plk2_catg_step;
                  }
              }

            plk1 += plk1_site_step;
            plk2 += plk2_site_step;

            if(scale_fast)
              {
                sum_scale_v1_val = (sum_scale_v1)?(sum_scale_v1[site]):(0);
                sum_scale_v2_val = (sum_scale_v2)?(sum_scale_v2[site]):(0);
                sum_scale[site] = sum_scale_v1_val + sum_scale_v2_val;

                if(do_scaling && largest_p_lk < INV_TWO_TO_THE_LARGE &&
                   tree->mod->augmented == NO &&
                   tree->apply_lk_scaling == YES)
                  {
                    plk0 -= ncatgns;
                    for(i=0;i<ncatgns;++i) plk0[i] *= TWO_TO_THE_LARGE;
                    sum_scale[site] += LARGE;
                    plk0 += ncatgns;
                  }
              }
          #else
            for(catg=0;catg<ncatg;++catg)
              {
                if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
                  {
                    SSE_Partial_Lk_Exex(_tPij1,state_v1,
                                        _tPij2,state_v2,
                                        ns,_plk0);
                  }
                else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
                  {
                    SSE_Partial_Lk_Exin(_tPij2,state_v2,
                                        _tPij1,plk1,_pmat1plk1,
                                        ns,_plk0);
                  }
                else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
                  {
                    SSE_Partial_Lk_Exin(_tPij1,state_v1,
                                        _tPij2,plk2,_pmat2plk2,
                                        ns,_plk0);
                  }
                else
                  {
                    SSE_Partial_Lk_Inin(_tPij1,plk1,_pmat1plk1,
                                        _tPij2,plk2,_pmat2plk2,
                                        ns,_plk0);
                  }

                for(k=0;k<nblocks;++k) _mm_store_pd(plk0+sz*k,_plk0[k]);
                _tPij1 += nsns / sz;
                _tPij2 += nsns / sz;
                plk0 += ns;
                plk1 += (n_v1->tax) ? 0 : ns;
                plk2 += (n_v2->tax) ? 0 : ns;
              }

            _tPij1 -= ncatg * nsns / sz;
            _tPij2 -= ncatg * nsns / sz;
            plk1 += (n_v1->tax) ? ns : 0;
            plk2 += (n_v2->tax) ? ns : 0;

            if(tree->scaling_method == SCALE_FAST)
              {
                sum_scale_v1_val = (sum_scale_v1)?(sum_scale_v1[site]):(0);
                sum_scale_v2_val = (sum_scale_v2)?(sum_scale_v2[site]):(0);
                sum_scale[site] = sum_scale_v1_val + sum_scale_v2_val;

                plk0 -= ncatgns;
                largest_p_lk = -BIG;
                for(i=0;i<ncatgns;++i)
                  if(plk0[i] > largest_p_lk)
                    largest_p_lk = plk0[i];

                if(largest_p_lk < INV_TWO_TO_THE_LARGE &&
                   tree->mod->augmented == NO &&
                   tree->apply_lk_scaling == YES)
                  {
                    for(i=0;i<ncatgns;++i) plk0[i] *= TWO_TO_THE_LARGE;
                    sum_scale[site] += LARGE;
                  }

                plk0 += ncatgns;
              }
          #endif
        }
      else
        {
          plk0 += ncatgns;
          plk1 += plk1_zero_wght_step;
          plk2 += plk2_zero_wght_step;        
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void SSE_Partial_Lk_Exex(const __m128d *_tPij1, const int state1, const __m128d *_tPij2, const int state2, const int ns, __m128d *plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  unsigned int i;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      SSE_Partial_Lk_Exex_4(_tPij1,state1,_tPij2,state2,plk0);
      return;
    }
  if(ns == 20)
    {
      SSE_Partial_Lk_Exex_20(_tPij1,state1,_tPij2,state2,plk0);
      return;
    }
#endif

  _tPij1 = _tPij1 + state1 * nblocks;
  _tPij2 = _tPij2 + state2 * nblocks;
  for(i=0;i<nblocks;++i) plk0[i] = _mm_mul_pd(_tPij1[i],_tPij2[i]);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void SSE_Partial_Lk_Exin(const __m128d *_tPij1, const int state1, const __m128d *_tPij2, const phydbl *_plk2, __m128d *_pmat2plk2, const int ns, __m128d *_plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  unsigned int i;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      SSE_Partial_Lk_Exin_4(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
      return;
    }
  if(ns == 20)
    {
      SSE_Partial_Lk_Exin_20(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
      return;
    }
#endif
  
  _tPij1 = _tPij1 + state1 * nblocks;
  SSE_Matrix_Vect_Prod(_tPij2,_plk2,ns,_pmat2plk2);
  
  for(i=0;i<nblocks;++i) _plk0[i] = _mm_mul_pd(_tPij1[i],_pmat2plk2[i]);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void SSE_Partial_Lk_Inin(const __m128d *_tPij1, const phydbl *plk1, __m128d *_pmat1plk1, const __m128d *_tPij2, const phydbl *plk2, __m128d *_pmat2plk2, const int ns, __m128d *_plk0)
{
  unsigned int i;
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      SSE_Partial_Lk_Inin_4(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
      return;
    }
  if(ns == 20)
    {
      SSE_Partial_Lk_Inin_20(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
      return;
    }
#endif
  
  for(i=0;i<ns;++i) if(plk1[i] > 1.0 || plk1[i] < 1.0 || plk2[i] > 1.0 || plk2[i] < 1.0) break; 

  if(i != ns)
    {     
      SSE_Matrix_Vect_Prod(_tPij1,plk1,ns,_pmat1plk1);
      SSE_Matrix_Vect_Prod(_tPij2,plk2,ns,_pmat2plk2);
      
      for(i=0;i<nblocks;++i) _plk0[i] = _mm_mul_pd(_pmat1plk1[i],_pmat2plk2[i]);
    }
  else
    {
      for(i=0;i<nblocks;++i) _plk0[i] = _mm_set1_pd(1.0);      
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void SSE_Matrix_Vect_Prod(const __m128d *_m_transpose, const phydbl *_v, const int ns, __m128d *_u)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;  
  unsigned int i,j;  
  __m128d _x;


  _x = _mm_set1_pd(_v[0]);
  for(j=0;j<nblocks;++j) _u[j] = _mm_mul_pd(_m_transpose[j],_x);
  _m_transpose = _m_transpose + nblocks;
  
  for(i=1;i<ns;++i)
    {
      _x = _mm_set1_pd(_v[i]);
      for(j=0;j<nblocks;++j)
        {
          _u[j] = _mm_add_pd(_u[j],_mm_mul_pd(_m_transpose[j],_x));
        }
      _m_transpose = _m_transpose + nblocks;
    }

  /* for(i=0;i<nblocks;++i) _u[i] = _mm_setzero_pd(); */

  /* for(i=0;i<ns;++i) */
  /*   { */
  /*     _x = _mm_set1_pd(_v[i]); */
  /*     for(j=0;j<nblocks;++j) _u[j] = _mm_add_pd(_u[j],_mm_mul_pd(_m_transpose[j],_x)); */
  /*     _m_transpose = _m_transpose + nblocks; */
  /*   } */
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

static inline phydbl SSE_Vect_Max(__m128d x)
{
  __m128d hi = _mm_unpackhi_pd(x, x);
  __m128d vmax = _mm_max_sd(x, hi);
  return _mm_cvtsd_f64(vmax);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

static inline phydbl SSE_Vects_Max(const __m128d *x, unsigned int nblocks)
{
  phydbl largest_p_lk = -BIG;
  unsigned int i;

  for(i=0;i<nblocks;++i)
    {
      const phydbl block_max = SSE_Vect_Max(x[i]);
      if(block_max > largest_p_lk) largest_p_lk = block_max;
    }

  return largest_p_lk;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#endif
