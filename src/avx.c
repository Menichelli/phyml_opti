/*

PhyML:  a program that  computes maximum likelihood phylogenies from
DNA or AA homologous sequences.

Copyright (C) Stephane Guindon. Oct 2003 onward.

All parts of the source except where indicated are distributed under
the GNU public licence. See http://www.opensource.org for details.

*/

#include "assert.h"
#include "avx.h"

phydbl Lk_Site_Eigen_Local(unsigned int site,
                           const phydbl *expl, const phydbl *dot_prod,
                           t_edge *b, t_tree *tree,
                           phydbl *site_lk_cat_local, int *site_warning);

#if ((defined(__AVX__) || defined(__AVX2__)) && !defined(DISABLE_NATIVE))

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

static inline phydbl AVX_Vect_Max(__m256d x);
static inline phydbl AVX_Vects_Max(const __m256d *x, unsigned int nblocks);

#if PHYML_MT_LK_RUNTIME
static int PhyML_MT_Force_Max_Threads_AVX(void)
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

static int PhyML_MT_Recommended_AVX_Threads(long long work, long long min_work_per_thread)
{
  const int max_threads = omp_get_max_threads();
  int threads;

  if(max_threads <= 1) return 1;
  if(PhyML_MT_Force_Max_Threads_AVX() == 1) return max_threads;
  if(max_threads <= 1 || work < min_work_per_thread) return 1;

  threads = (int)((work + min_work_per_thread - 1LL) / min_work_per_thread);
  if(threads < 1) threads = 1;
  if(threads > max_threads) threads = max_threads;
  return threads;
}

static int PhyML_MT_Threads_AVX_Update_Partial_Lk(int npatterns, int ncatg, int ns)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns;
  return PhyML_MT_Recommended_AVX_Threads(work,320000LL);
}

static int PhyML_MT_Threads_AVX_Update_Eigen_Lr(int npatterns, int ncatg, int ns)
{
  const long long work = (long long)npatterns * (long long)ncatg * (long long)ns * (long long)ns;
  return PhyML_MT_Recommended_AVX_Threads(work,160000LL);
}

static void PhyML_MT_AVX_Get_Site_Range(unsigned int nsites, unsigned int *begin, unsigned int *end)
{
  const unsigned int tid = (unsigned int)omp_get_thread_num();
  const unsigned int nth = (unsigned int)omp_get_num_threads();

  *begin = (unsigned int)(((unsigned long long)nsites * tid) / nth);
  *end   = (unsigned int)(((unsigned long long)nsites * (tid + 1U)) / nth);
}
#endif

static void AVX_Prepare_Eigen_Packs(t_tree *tree)
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
          tree->_r_ev[i*nblocks+j] = _mm256_load_pd(r_ev + j*sz);
          tree->_l_ev[i*nblocks+j] = _mm256_load_pd(l_ev + j*sz);
        }
      r_ev += ns;
      l_ev += ns;
    }

  tree->eigen_pack_epoch = tree->mod->eigen_epoch;
  tree->eigen_pack_valid = YES;
}

static const __m256d *AVX_Find_Packed_tPij(const t_tree *tree, const phydbl *raw_tPij)
{
  int i;

  if(raw_tPij == NULL) return NULL;

  for(i=0;i<2*tree->n_otu-1;++i)
    {
      const t_edge *edge = tree->a_edges[i];

      if(edge != NULL && edge->tPij_rr == raw_tPij)
        {
          assert(edge->packed_tPij_rr != NULL);
          return (const __m256d *)edge->packed_tPij_rr;
        }
    }

  assert(FALSE);
  return NULL;
}

#if PHYML_OPT_PARTIAL_LK
static inline int AVX_All_One(const phydbl *plk, unsigned int ns)
{
  unsigned int i;
  for(i=0;i<ns;++i) if(plk[i] != 1.0) return 0;
  return 1;
}

static inline void AVX_Matrix_Vect_Prod_4(const __m256d *_m_transpose, const phydbl *_v, __m256d *_u)
{
  const __m256d x0 = _mm256_set1_pd(_v[0]);
  const __m256d x1 = _mm256_set1_pd(_v[1]);
  const __m256d x2 = _mm256_set1_pd(_v[2]);
  const __m256d x3 = _mm256_set1_pd(_v[3]);

#if (defined(__FMA__))
  _u[0] = _mm256_fmadd_pd(_m_transpose[3],x3,
          _mm256_fmadd_pd(_m_transpose[2],x2,
          _mm256_fmadd_pd(_m_transpose[1],x1,
                          _mm256_mul_pd(_m_transpose[0],x0))));
#else
  _u[0] = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_m_transpose[0],x0),
                                      _mm256_mul_pd(_m_transpose[1],x1)),
                        _mm256_add_pd(_mm256_mul_pd(_m_transpose[2],x2),
                                      _mm256_mul_pd(_m_transpose[3],x3)));
#endif
}

static inline void AVX_Matrix_Vect_Prod_20(const __m256d *_m_transpose, const phydbl *_v, __m256d *_u)
{
  unsigned int i;
  __m256d x;

  x = _mm256_set1_pd(_v[0]);
  _u[0] = _mm256_mul_pd(_m_transpose[0],x);
  _u[1] = _mm256_mul_pd(_m_transpose[1],x);
  _u[2] = _mm256_mul_pd(_m_transpose[2],x);
  _u[3] = _mm256_mul_pd(_m_transpose[3],x);
  _u[4] = _mm256_mul_pd(_m_transpose[4],x);

  for(i=1;i<20;++i)
    {
      const __m256d *row = _m_transpose + 5*i;
      x = _mm256_set1_pd(_v[i]);
#if (defined(__FMA__))
      _u[0] = _mm256_fmadd_pd(row[0],x,_u[0]);
      _u[1] = _mm256_fmadd_pd(row[1],x,_u[1]);
      _u[2] = _mm256_fmadd_pd(row[2],x,_u[2]);
      _u[3] = _mm256_fmadd_pd(row[3],x,_u[3]);
      _u[4] = _mm256_fmadd_pd(row[4],x,_u[4]);
#else
      _u[0] = _mm256_add_pd(_u[0],_mm256_mul_pd(row[0],x));
      _u[1] = _mm256_add_pd(_u[1],_mm256_mul_pd(row[1],x));
      _u[2] = _mm256_add_pd(_u[2],_mm256_mul_pd(row[2],x));
      _u[3] = _mm256_add_pd(_u[3],_mm256_mul_pd(row[3],x));
      _u[4] = _mm256_add_pd(_u[4],_mm256_mul_pd(row[4],x));
#endif
    }
}

static inline void AVX_Partial_Lk_Exex_4(const __m256d *_tPij1, const int state1,
                                         const __m256d *_tPij2, const int state2,
                                         __m256d *plk0)
{
  plk0[0] = _mm256_mul_pd(_tPij1[state1],_tPij2[state2]);
}

static inline void AVX_Partial_Lk_Exex_20(const __m256d *_tPij1, const int state1,
                                          const __m256d *_tPij2, const int state2,
                                          __m256d *plk0)
{
  const __m256d *col1 = _tPij1 + 5*state1;
  const __m256d *col2 = _tPij2 + 5*state2;

  plk0[0] = _mm256_mul_pd(col1[0],col2[0]);
  plk0[1] = _mm256_mul_pd(col1[1],col2[1]);
  plk0[2] = _mm256_mul_pd(col1[2],col2[2]);
  plk0[3] = _mm256_mul_pd(col1[3],col2[3]);
  plk0[4] = _mm256_mul_pd(col1[4],col2[4]);
}

static inline phydbl AVX_Partial_Lk_Exex_4_Max(const __m256d *_tPij1, const int state1,
                                               const __m256d *_tPij2, const int state2,
                                               __m256d *plk0)
{
  AVX_Partial_Lk_Exex_4(_tPij1,state1,_tPij2,state2,plk0);
  return AVX_Vect_Max(plk0[0]);
}

static inline phydbl AVX_Partial_Lk_Exex_20_Max(const __m256d *_tPij1, const int state1,
                                                const __m256d *_tPij2, const int state2,
                                                __m256d *plk0)
{
  phydbl largest_p_lk;

  AVX_Partial_Lk_Exex_20(_tPij1,state1,_tPij2,state2,plk0);
  largest_p_lk = AVX_Vect_Max(plk0[0]);
  if(AVX_Vect_Max(plk0[1]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(plk0[1]);
  if(AVX_Vect_Max(plk0[2]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(plk0[2]);
  if(AVX_Vect_Max(plk0[3]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(plk0[3]);
  if(AVX_Vect_Max(plk0[4]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(plk0[4]);
  return largest_p_lk;
}

static inline void AVX_Partial_Lk_Exin_4(const __m256d *_tPij1, const int state1,
                                         const __m256d *_tPij2, const phydbl *_plk2,
                                         __m256d *_pmat2plk2, __m256d *_plk0)
{
  AVX_Matrix_Vect_Prod_4(_tPij2,_plk2,_pmat2plk2);
  _plk0[0] = _mm256_mul_pd(_tPij1[state1],_pmat2plk2[0]);
}

static inline void AVX_Partial_Lk_Exin_20(const __m256d *_tPij1, const int state1,
                                          const __m256d *_tPij2, const phydbl *_plk2,
                                          __m256d *_pmat2plk2, __m256d *_plk0)
{
  const __m256d *col1 = _tPij1 + 5*state1;

  AVX_Matrix_Vect_Prod_20(_tPij2,_plk2,_pmat2plk2);
  _plk0[0] = _mm256_mul_pd(col1[0],_pmat2plk2[0]);
  _plk0[1] = _mm256_mul_pd(col1[1],_pmat2plk2[1]);
  _plk0[2] = _mm256_mul_pd(col1[2],_pmat2plk2[2]);
  _plk0[3] = _mm256_mul_pd(col1[3],_pmat2plk2[3]);
  _plk0[4] = _mm256_mul_pd(col1[4],_pmat2plk2[4]);
}

static inline phydbl AVX_Partial_Lk_Exin_4_Max(const __m256d *_tPij1, const int state1,
                                               const __m256d *_tPij2, const phydbl *_plk2,
                                               __m256d *_pmat2plk2, __m256d *_plk0)
{
  AVX_Partial_Lk_Exin_4(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
  return AVX_Vect_Max(_plk0[0]);
}

static inline phydbl AVX_Partial_Lk_Exin_20_Max(const __m256d *_tPij1, const int state1,
                                                const __m256d *_tPij2, const phydbl *_plk2,
                                                __m256d *_pmat2plk2, __m256d *_plk0)
{
  phydbl largest_p_lk;

  AVX_Partial_Lk_Exin_20(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
  largest_p_lk = AVX_Vect_Max(_plk0[0]);
  if(AVX_Vect_Max(_plk0[1]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[1]);
  if(AVX_Vect_Max(_plk0[2]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[2]);
  if(AVX_Vect_Max(_plk0[3]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[3]);
  if(AVX_Vect_Max(_plk0[4]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[4]);
  return largest_p_lk;
}

static inline void AVX_Partial_Lk_Inin_4(const __m256d *_tPij1, const phydbl *plk1,
                                         __m256d *_pmat1plk1, const __m256d *_tPij2,
                                         const phydbl *plk2, __m256d *_pmat2plk2,
                                         __m256d *_plk0)
{
  if(AVX_All_One(plk1,4) && AVX_All_One(plk2,4))
    {
      _plk0[0] = _mm256_set1_pd(1.0);
      return;
    }

  AVX_Matrix_Vect_Prod_4(_tPij1,plk1,_pmat1plk1);
  AVX_Matrix_Vect_Prod_4(_tPij2,plk2,_pmat2plk2);
  _plk0[0] = _mm256_mul_pd(_pmat1plk1[0],_pmat2plk2[0]);
}

static void AVX_Partial_Lk_Inin_20(const __m256d *_tPij1, const phydbl *plk1,
                                   __m256d *_pmat1plk1, const __m256d *_tPij2,
                                   const phydbl *plk2, __m256d *_pmat2plk2,
                                   __m256d *_plk0)
{
  __m256d u2[5];

  (void)_pmat1plk1;
  (void)_pmat2plk2;

  if(AVX_All_One(plk1,20) && AVX_All_One(plk2,20))
    {
      _plk0[0] = _mm256_set1_pd(1.0);
      _plk0[1] = _mm256_set1_pd(1.0);
      _plk0[2] = _mm256_set1_pd(1.0);
      _plk0[3] = _mm256_set1_pd(1.0);
      _plk0[4] = _mm256_set1_pd(1.0);
      return;
    }

  AVX_Matrix_Vect_Prod_20(_tPij1,plk1,_plk0);
  AVX_Matrix_Vect_Prod_20(_tPij2,plk2,u2);

  _plk0[0] = _mm256_mul_pd(_plk0[0],u2[0]);
  _plk0[1] = _mm256_mul_pd(_plk0[1],u2[1]);
  _plk0[2] = _mm256_mul_pd(_plk0[2],u2[2]);
  _plk0[3] = _mm256_mul_pd(_plk0[3],u2[3]);
  _plk0[4] = _mm256_mul_pd(_plk0[4],u2[4]);
}

static inline phydbl AVX_Partial_Lk_Inin_4_Max(const __m256d *_tPij1, const phydbl *plk1,
                                               __m256d *_pmat1plk1, const __m256d *_tPij2,
                                               const phydbl *plk2, __m256d *_pmat2plk2,
                                               __m256d *_plk0)
{
  AVX_Partial_Lk_Inin_4(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
  return AVX_Vect_Max(_plk0[0]);
}

static inline phydbl AVX_Partial_Lk_Inin_20_Max(const __m256d *_tPij1, const phydbl *plk1,
                                                __m256d *_pmat1plk1, const __m256d *_tPij2,
                                                const phydbl *plk2, __m256d *_pmat2plk2,
                                                __m256d *_plk0)
{
  phydbl largest_p_lk;

  AVX_Partial_Lk_Inin_20(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
  largest_p_lk = AVX_Vect_Max(_plk0[0]);
  if(AVX_Vect_Max(_plk0[1]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[1]);
  if(AVX_Vect_Max(_plk0[2]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[2]);
  if(AVX_Vect_Max(_plk0[3]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[3]);
  if(AVX_Vect_Max(_plk0[4]) > largest_p_lk) largest_p_lk = AVX_Vect_Max(_plk0[4]);
  return largest_p_lk;
}

static inline phydbl AVX_Partial_Lk_Exex_Max(const __m256d *_tPij1, const int state1,
                                             const __m256d *_tPij2, const int state2,
                                             const int ns, __m256d *plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  phydbl largest_p_lk = -BIG;
  unsigned int i;

  if(ns == 4) return AVX_Partial_Lk_Exex_4_Max(_tPij1,state1,_tPij2,state2,plk0);
  if(ns == 20) return AVX_Partial_Lk_Exex_20_Max(_tPij1,state1,_tPij2,state2,plk0);

  _tPij1 = _tPij1 + state1 * nblocks;
  _tPij2 = _tPij2 + state2 * nblocks;
  for(i=0;i<nblocks;++i)
    {
      const __m256d x = _mm256_mul_pd(_tPij1[i],_tPij2[i]);
      const phydbl block_max = AVX_Vect_Max(x);
      plk0[i] = x;
      if(block_max > largest_p_lk) largest_p_lk = block_max;
    }

  return largest_p_lk;
}

static inline phydbl AVX_Partial_Lk_Exin_Max(const __m256d *_tPij1, const int state1,
                                             const __m256d *_tPij2, const phydbl *_plk2,
                                             __m256d *_pmat2plk2, const int ns, __m256d *_plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  phydbl largest_p_lk = -BIG;
  unsigned int i;

  if(ns == 4) return AVX_Partial_Lk_Exin_4_Max(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
  if(ns == 20) return AVX_Partial_Lk_Exin_20_Max(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);

  _tPij1 = _tPij1 + state1 * nblocks;
  AVX_Matrix_Vect_Prod(_tPij2,_plk2,ns,_pmat2plk2);

  for(i=0;i<nblocks;++i)
    {
      const __m256d x = _mm256_mul_pd(_tPij1[i],_pmat2plk2[i]);
      const phydbl block_max = AVX_Vect_Max(x);
      _plk0[i] = x;
      if(block_max > largest_p_lk) largest_p_lk = block_max;
    }

  return largest_p_lk;
}

static inline phydbl AVX_Partial_Lk_Inin_Max(const __m256d *_tPij1, const phydbl *plk1,
                                             __m256d *_pmat1plk1, const __m256d *_tPij2,
                                             const phydbl *plk2, __m256d *_pmat2plk2,
                                             const int ns, __m256d *_plk0)
{
  unsigned int i;
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  phydbl largest_p_lk = -BIG;

  if(ns == 4) return AVX_Partial_Lk_Inin_4_Max(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
  if(ns == 20) return AVX_Partial_Lk_Inin_20_Max(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);

  for(i=0;i<ns;++i) if(plk1[i] > 1.0 || plk1[i] < 1.0 || plk2[i] > 1.0 || plk2[i] < 1.0) break;

  if(i != ns)
    {
      AVX_Matrix_Vect_Prod(_tPij1,plk1,ns,_pmat1plk1);
      AVX_Matrix_Vect_Prod(_tPij2,plk2,ns,_pmat2plk2);

      for(i=0;i<nblocks;++i)
        {
          const __m256d x = _mm256_mul_pd(_pmat1plk1[i],_pmat2plk2[i]);
          const phydbl block_max = AVX_Vect_Max(x);
          _plk0[i] = x;
          if(block_max > largest_p_lk) largest_p_lk = block_max;
        }
    }
  else
    {
      for(i=0;i<nblocks;++i) _plk0[i] = _mm256_set1_pd(1.0);
      largest_p_lk = 1.0;
    }

  return largest_p_lk;
}

#if PHYML_MT_LK_RUNTIME
static void AVX_Copy_tPij_Local(const phydbl *src_tPij1, const phydbl *src_tPij2,
                                __m256d *dst_tPij1, __m256d *dst_tPij2,
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
              dst_tPij1[k] = _mm256_load_pd(src_tPij1);
              dst_tPij2[k] = _mm256_load_pd(src_tPij2);
              src_tPij1 += sz;
              src_tPij2 += sz;
            }
          dst_tPij1 += nblocks;
          dst_tPij2 += nblocks;
        }
    }
}

static void AVX_Update_Partial_Lk_Prepared_Range(t_tree *tree,
                                                 const t_node *n_v1, const t_node *n_v2,
                                                 phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                 int *sum_scale, const int *sum_scale_v1, const int *sum_scale_v2,
                                                 const __m256d *init_tPij1, const __m256d *init_tPij2,
                                                 const unsigned int site_begin, const unsigned int site_end,
                                                 const unsigned int ns, const unsigned int ncatg,
                                                 const phydbl *wght,
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
  __m256d *pmat1plk1_local;
  __m256d *pmat2plk2_local;
  __m256d *plk0_local;
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
      const __m256d *site_tPij1,*site_tPij2;

      if(wght[site] <= SMALL) continue;

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
                  catg_largest_p_lk = AVX_Partial_Lk_Exex_Max(site_tPij1,state_v1,
                                                              site_tPij2,state_v2,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  AVX_Partial_Lk_Exex(site_tPij1,state_v1,site_tPij2,state_v2,ns,plk0_local);
                }
            }
          else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
            {
              if(do_scaling)
                {
                  catg_largest_p_lk = AVX_Partial_Lk_Exin_Max(site_tPij2,state_v2,
                                                              site_tPij1,catg_plk1,pmat1plk1_local,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  AVX_Partial_Lk_Exin(site_tPij2,state_v2,
                                      site_tPij1,catg_plk1,pmat1plk1_local,
                                      ns,plk0_local);
                }
            }
          else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
            {
              if(do_scaling)
                {
                  catg_largest_p_lk = AVX_Partial_Lk_Exin_Max(site_tPij1,state_v1,
                                                              site_tPij2,catg_plk2,pmat2plk2_local,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  AVX_Partial_Lk_Exin(site_tPij1,state_v1,
                                      site_tPij2,catg_plk2,pmat2plk2_local,
                                      ns,plk0_local);
                }
            }
          else
            {
              if(do_scaling)
                {
                  catg_largest_p_lk = AVX_Partial_Lk_Inin_Max(site_tPij1,catg_plk1,pmat1plk1_local,
                                                              site_tPij2,catg_plk2,pmat2plk2_local,
                                                              ns,plk0_local);
                  if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                }
              else
                {
                  AVX_Partial_Lk_Inin(site_tPij1,catg_plk1,pmat1plk1_local,
                                      site_tPij2,catg_plk2,pmat2plk2_local,
                                      ns,plk0_local);
                }
            }

          for(k=0;k<nblocks;++k) _mm256_store_pd(catg_plk0 + sz*k,plk0_local[k]);
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

static void AVX_Update_Partial_Lk_Prepared_Team(t_tree *tree,
                                                const t_node *n_v1, const t_node *n_v2,
                                                phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                                int *sum_scale, const int *sum_scale_v1, const int *sum_scale_v2,
                                                const __m256d *init_tPij1, const __m256d *init_tPij2,
                                                const unsigned int npattern, const unsigned int ns, const unsigned int ncatg,
                                                const phydbl *wght,
                                                t_lk_thread_ctx *ctx)
{
  unsigned int begin,end;

  assert(ctx != NULL);

  PhyML_MT_AVX_Get_Site_Range(npattern,&begin,&end);
  AVX_Update_Partial_Lk_Prepared_Range(tree,
                                       n_v1,n_v2,
                                       plk0,plk1,plk2,
                                       sum_scale,sum_scale_v1,sum_scale_v2,
                                       init_tPij1,init_tPij2,
                                       begin,end,ns,ncatg,wght,ctx);
}

static void AVX_Update_Partial_Lk_MT(t_tree *tree,
                                     const t_node *n_v1, const t_node *n_v2,
                                     phydbl *plk0, const phydbl *plk1, const phydbl *plk2,
                                     int *sum_scale, const int *sum_scale_v1, const int *sum_scale_v2,
                                     const __m256d *init_tPij1, const __m256d *init_tPij2,
                                     const unsigned int npattern, const unsigned int ns, const unsigned int ncatg,
                                     const phydbl *wght,
                                     const int nthreads)
{
  #pragma omp parallel num_threads(nthreads)
    {
      t_lk_thread_ctx *ctx = tree->lk_thread_ctx + omp_get_thread_num();
      AVX_Update_Partial_Lk_Prepared_Team(tree,
                                          n_v1,n_v2,
                                          plk0,plk1,plk2,
                                          sum_scale,sum_scale_v1,sum_scale_v2,
                                          init_tPij1,init_tPij2,
                                          npattern,ns,ncatg,wght,ctx);
    }
}
#endif
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#if PHYML_MT_LK_RUNTIME
#if PHYML_OPT_PARTIAL_LK
void AVX_Update_Partial_Lk_Team(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx)
{
  t_node *n_v1, *n_v2;
  phydbl *plk0,*plk1,*plk2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;
  const phydbl *wght;
  int use_alias;
  const unsigned int npattern = tree->n_pattern;
  const unsigned int ns = tree->mod->ns;
  const unsigned int ncatg = tree->mod->ras->n_catg;
  const __m256d *init_tPij1,*init_tPij2;
  unsigned int begin,end;

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
        PhyML_Printf("\n== AVX version of the Update_Partial_Lk function does not");
        PhyML_Printf("\n== allow augmented data.");
        assert(FALSE);
      }
      return;
    }

  init_tPij1 = AVX_Find_Packed_tPij(tree,tPij1);
  init_tPij2 = AVX_Find_Packed_tPij(tree,tPij2);
  wght = tree->data->wght;
  use_alias = PhyML_Use_Subpatt_Aliasing(tree,p_lk_loc);

  if(use_alias == YES)
    {
      #pragma omp single
      {
        PhyML_Prepare_Subpatt_Weight_Mask(tree,p_lk_loc);
      }
      #pragma omp barrier
      if(tree->alias_subpatt_nactive_sites < npattern)
        wght = tree->alias_subpatt_wght;
      else
        use_alias = NO;
    }

  AVX_Update_Partial_Lk_Prepared_Team(tree,
                                      n_v1,n_v2,
                                      plk0,plk1,plk2,
                                      sum_scale,sum_scale_v1,sum_scale_v2,
                                      init_tPij1,init_tPij2,
                                      npattern,ns,ncatg,wght,ctx);

  if(use_alias == YES)
    {
      #pragma omp barrier
      PhyML_MT_AVX_Get_Site_Range(npattern,&begin,&end);
      PhyML_Copy_Subpatt_Partials_Range(tree,plk0,sum_scale,p_lk_loc,ncatg,ns,begin,end);
      #pragma omp barrier
    }
}

void AVX_Update_Partial_Lk_Wavefront_Job(t_tree *tree, t_edge *b, t_node *d, t_lk_thread_ctx *ctx)
{
  t_node *n_v1, *n_v2;
  phydbl *plk0,*plk1,*plk2;
  phydbl *Pij1,*Pij2;
  phydbl *tPij1,*tPij2;
  int *sum_scale, *sum_scale_v1, *sum_scale_v2;
  int *p_lk_loc;
  const phydbl *wght;
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
      PhyML_Printf("\n== AVX version of the Update_Partial_Lk function does not");
      PhyML_Printf("\n== allow augmented data.");
      assert(FALSE);
    }

  wght = PhyML_Prepare_Subpatt_Weight_Mask(tree,p_lk_loc);

  AVX_Update_Partial_Lk_Prepared_Range(tree,
                                       n_v1,n_v2,
                                       plk0,plk1,plk2,
                                       sum_scale,sum_scale_v1,sum_scale_v2,
                                       AVX_Find_Packed_tPij(tree,tPij1),
                                       AVX_Find_Packed_tPij(tree,tPij2),
                                       0U,npattern,ns,ncatg,wght,ctx);

  if(wght != tree->data->wght)
    PhyML_Copy_Subpatt_Partials(tree,plk0,sum_scale,p_lk_loc,ncatg,ns);
}
#endif

static void AVX_Update_Eigen_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx, int fused_site_lk)
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
  __m256d *_l_ev,*_r_ev;
  phydbl *p_lk_left_pi;
  __m256d *prod_left,*prod_rght;

  assert(ctx != NULL);
  assert(sz == 4);
  assert(tree->update_eigen_lr == YES);

  p_lk_left_pi = ctx->p_lk_left_pi;
  prod_left    = ctx->_prod_left;
  prod_rght    = ctx->_prod_rght;
  _l_ev        = tree->_l_ev;
  _r_ev        = tree->_r_ev;
  pi           = tree->mod->e_frq->pi->v;

  #pragma omp single
  AVX_Prepare_Eigen_Packs(tree);

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

              AVX_Matrix_Vect_Prod(_r_ev,p_lk_left_pi,ns,prod_left);
              AVX_Matrix_Vect_Prod(_l_ev,site_p_lk_rght,ns,prod_rght);

              for(i=0;i<nblocks;++i)
                _mm256_store_pd(site_dot_prod + i*sz,_mm256_mul_pd(prod_left[i],prod_rght[i]));

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

void AVX_Update_Eigen_Lr_Team(t_edge *b, t_tree *tree, t_lk_thread_ctx *ctx)
{
  AVX_Update_Eigen_Sites_Team(b,tree,NULL,ctx,NO);
}

void AVX_Update_Eigen_And_Lk_Sites_Team(t_edge *b, t_tree *tree, const phydbl *expl, t_lk_thread_ctx *ctx)
{
  AVX_Update_Eigen_Sites_Team(b,tree,expl,ctx,YES);
}
#endif

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void AVX_Update_Eigen_Lr(t_edge *b, t_tree *tree)
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

  __m256d *_l_ev,*_r_ev,*_prod_left,*_prod_rght;

  p_lk_left_pi = tree->p_lk_left_pi;
  _l_ev        = tree->_l_ev;
  _r_ev        = tree->_r_ev;
  _prod_left   = tree->_prod_left;
  _prod_rght   = tree->_prod_rght;
    
  assert(sz == 4);
  assert(tree->update_eigen_lr == YES);

  AVX_Prepare_Eigen_Packs(tree);
  
  p_lk_left = b->left->tax ? b->p_lk_tip_l : b->p_lk_left;
  p_lk_rght = b->rght->tax ? b->p_lk_tip_r : b->p_lk_rght;
  pi = tree->mod->e_frq->pi->v;
  dot_prod = tree->dot_prod;

#if PHYML_MT_LK_RUNTIME
  {
    const int nthreads = PhyML_MT_Threads_AVX_Update_Eigen_Lr((int)npattern,(int)ncatg,(int)ns);

    if(nthreads > 1 && tree->lk_thread_ctx != NULL)
      {
        #pragma omp parallel num_threads(nthreads)
        {
          t_lk_thread_ctx *ctx = tree->lk_thread_ctx + omp_get_thread_num();
          AVX_Update_Eigen_Lr_Team(b,tree,ctx);
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
              
              AVX_Matrix_Vect_Prod(_r_ev,p_lk_left_pi,ns,_prod_left);
              AVX_Matrix_Vect_Prod(_l_ev,p_lk_rght,ns,_prod_rght);
              
              for(i=0;i<nblocks;++i) _mm256_store_pd(dot_prod + i*sz,_mm256_mul_pd(_prod_left[i],_prod_rght[i]));
              
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

phydbl AVX_Lk_Core_One_Class_No_Eigen_Lr(const phydbl *p_lk_left, const phydbl *p_lk_rght, const phydbl *Pij, const phydbl *tPij, const phydbl *pi, const int ns, const int ambiguity_check, const int observed_state)
{
  const unsigned int sz = (int)BYTE_ALIGN / 8;
  const unsigned nblocks = ns/sz;

  if(nblocks == 1)
    {
      __m256d _plk_r,_plk_l;

      if(ambiguity_check == NO) // tip case.
        {
          Pij += observed_state*ns;
          _plk_r = _mm256_mul_pd(_mm256_load_pd(Pij),_mm256_load_pd(p_lk_left));
          return pi[observed_state] * AVX_Vect_Norm(_plk_r);
        }
      else
        {
          unsigned int i;
          __m256d _pijplk,_pij;
          
          _plk_r  = _mm256_mul_pd(_mm256_load_pd(p_lk_rght),_mm256_load_pd(pi));
          _pijplk = _mm256_setzero_pd();
          
          for(i=0;i<ns;++i)
            {
              _pij = _mm256_load_pd(tPij);
              
              
#if (defined(__FMA__))
              _pijplk = _mm256_fmadd_pd(_pij,_mm256_set1_pd(p_lk_left[i]),_pijplk);
#else
              _pijplk = _mm256_add_pd(_pijplk,_mm256_mul_pd(_pij,_mm256_set1_pd(p_lk_left[i])));
#endif
              tPij += ns;
            }
          
          _plk_l = _mm256_mul_pd(_pijplk,_plk_r);      
          
          return(AVX_Vect_Norm(_plk_l));
        }
      return UNLIKELY;
    }
  else
    {
      __m256d _plk_l[nblocks],_plk_r[nblocks];
      __m256d _plk;

      /* [ Pi . Lkr ]' x Pij x Lkl */
      
      if(ambiguity_check == NO) // tip case.
        {
          unsigned int i;
          
          Pij += observed_state*ns;
          
          for(i=0;i<nblocks;++i)
            {
              _plk_l[i] = _mm256_load_pd(p_lk_left);
              _plk_r[i] = _mm256_load_pd(Pij);
              _plk_r[i] = _mm256_mul_pd(_plk_r[i],_plk_l[i]);
              p_lk_left += sz;
              Pij += sz;
            }
          
          _plk = _mm256_setzero_pd();
          for(i=0;i<nblocks;++i) _plk = _mm256_add_pd(_plk,_plk_r[i]);
          return pi[observed_state] * AVX_Vect_Norm(_plk);
        }
      else
        {
          unsigned int i,j;
          __m256d _pij[nblocks],_pijplk[nblocks];
          phydbl lk;
          
          for(i=0;i<nblocks;++i)
            {
              _plk_r[i] = _mm256_mul_pd(_mm256_load_pd(p_lk_rght),_mm256_load_pd(pi));
              p_lk_rght += sz;
              pi += sz;
            }
          
          for(i=0;i<nblocks;++i) _pijplk[i] = _mm256_setzero_pd();
          
          for(i=0;i<ns;++i)
            {
              for(j=0;j<nblocks;++j)
                {
                  _pij[j] = _mm256_load_pd(tPij);
                  tPij += sz;
                  
#if (defined(__FMA__))
                  _pijplk[j] = _mm256_fmadd_pd(_pij[j],_mm256_set1_pd(p_lk_left[i]),_pijplk[j]);
#else
                  _pijplk[j] = _mm256_add_pd(_pijplk[j],_mm256_mul_pd(_pij[j],_mm256_set1_pd(p_lk_left[i])));
#endif
                }
            }
          
          lk = 0.0;
          for(i=0;i<nblocks;++i) lk += AVX_Vect_Norm(_mm256_mul_pd(_pijplk[i],_plk_r[i]));
          return lk;
        }
      
      return UNLIKELY;
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl AVX_Lk_Core_One_Class_Eigen_Lr(const phydbl *dot_prod, const phydbl *expl, const unsigned int ns)
{
  unsigned int l;
  const unsigned int sz = (int)BYTE_ALIGN / 8;
  const unsigned nblocks = ns/sz;
  __m256d _prod[nblocks],_x;
  
  for(l=0;l<nblocks;++l)
    {
      _prod[l] = _mm256_load_pd(dot_prod); dot_prod += sz;
    }

  if(expl != NULL)
    {
      for(l=0;l<nblocks;++l)
        {
          _prod[l] = _mm256_mul_pd(_prod[l],_mm256_load_pd(expl));
          expl += sz;
        }
    }
  
  _x = _mm256_setzero_pd();
  for(l=0;l<nblocks;++l) _x = _mm256_add_pd(_x,_prod[l]);
  
  return AVX_Vect_Norm(_x);
}
 
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void AVX_Lk_dLk_Core_One_Class_Eigen_Lr(const phydbl *dot_prod, const phydbl *expl, const unsigned int ns, phydbl *lk, phydbl *dlk)
{
  unsigned int i;
  const unsigned int sz = (int)BYTE_ALIGN / 8;
  const unsigned nblocks = ns/sz*2;
  __m256d _x,_y,_z;

  _z = _mm256_setzero_pd();

  for(i=0;i<nblocks;++i)
    {
      _x = _mm256_blend_pd(_mm256_set1_pd(dot_prod[2*i]),
                           _mm256_set1_pd(dot_prod[2*i + 1]),12);
      
      _y = _mm256_load_pd(expl + 4*i);

#if (defined(__FMA__))
      _z = _mm256_fmadd_pd(_x,_y,_z);
#else
      _z = _mm256_add_pd(_z,_mm256_mul_pd(_x,_y));
#endif
    }
  
  *lk = ((double *)&_z)[0] + ((double *)&_z)[2];
  *dlk = ((double *)&_z)[1] + ((double *)&_z)[3];

}
 
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

phydbl AVX_Vect_Norm(__m256d _z)
{
  __m128d vlow  = _mm256_castpd256_pd128(_z);
  __m128d vhigh = _mm256_extractf128_pd(_z, 1); // high 128
  vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
  
  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

      
  /* phydbl r; */
  /* __m256d _x = _mm256_hadd_pd(_z,_z); */
  /* __m256d _y = _mm256_permute2f128_pd(_x,_x,0x21); */
  /* _mm_store_sd(&r,_mm256_castpd256_pd128(_mm256_add_pd(_x,_y))); */
  /* return r; */
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void AVX_Update_Partial_Lk(t_tree *tree, t_edge *b, t_node *d)
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
  phydbl *plk0_base;
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

  __m256d *_tPij1,*_tPij2,*_pmat1plk1,*_pmat2plk2,*_plk0;
  const __m256d *init_tPij1,*init_tPij2;
  int tax_v1, tax_v2;
  const phydbl *wght;
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
  plk0_base = plk0;

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


  /* PhyML_Printf("\n. b: %d b->left:%d b->rght:%d d:%d [%p,%p]", */
  /*              b->num, */
  /*              b->left->num, */
  /*              b->rght->num, */
  /*              d->num, */
  /*              plk0, */
  /*              b->p_lk_rght); */

  init_tPij1 = AVX_Find_Packed_tPij(tree,tPij1);
  init_tPij2 = AVX_Find_Packed_tPij(tree,tPij2);
  wght = PhyML_Prepare_Subpatt_Weight_Mask(tree,p_lk_loc);

  if(tree->mod->augmented == YES)
    {
      PhyML_Printf("\n== AVX version of the Update_Partial_Lk function does not");
      PhyML_Printf("\n== allow augmented data.");
      assert(FALSE);
    }

#if PHYML_MT_LK_RUNTIME && PHYML_OPT_PARTIAL_LK
  {
    const int nthreads = PhyML_MT_Threads_AVX_Update_Partial_Lk((int)npattern,(int)ncatg,(int)ns);

    if(nthreads > 1)
    {
      AVX_Update_Partial_Lk_MT(tree,
                               n_v1,n_v2,
                               plk0,plk1,plk2,
                               sum_scale,sum_scale_v1,sum_scale_v2,
                               init_tPij1,init_tPij2,
                               npattern,ns,ncatg,wght,nthreads);
      if(wght != tree->data->wght)
        PhyML_Copy_Subpatt_Partials(tree,plk0_base,sum_scale,p_lk_loc,ncatg,ns);
      return;
    }
  }
#endif
    
  /* For every site in the alignment */
  for(site=0;site<npattern;++site)
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

          #if PHYML_OPT_PARTIAL_LK
            _tPij1 = (__m256d *)init_tPij1;
            _tPij2 = (__m256d *)init_tPij2;
            if(do_scaling) largest_p_lk = -BIG;

            if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
              {
                for(catg=0;catg<ncatg;++catg)
                  {
                    if(do_scaling)
                      {
                        catg_largest_p_lk = AVX_Partial_Lk_Exex_Max(_tPij1,state_v1,
                                                                    _tPij2,state_v2,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        AVX_Partial_Lk_Exex(_tPij1,state_v1,
                                            _tPij2,state_v2,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm256_store_pd(plk0+sz*k,_plk0[k]);

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
                        catg_largest_p_lk = AVX_Partial_Lk_Exin_Max(_tPij2,state_v2,
                                                                    _tPij1,plk1,_pmat1plk1,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        AVX_Partial_Lk_Exin(_tPij2,state_v2,
                                            _tPij1,plk1,_pmat1plk1,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm256_store_pd(plk0+sz*k,_plk0[k]);

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
                        catg_largest_p_lk = AVX_Partial_Lk_Exin_Max(_tPij1,state_v1,
                                                                    _tPij2,plk2,_pmat2plk2,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        AVX_Partial_Lk_Exin(_tPij1,state_v1,
                                            _tPij2,plk2,_pmat2plk2,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm256_store_pd(plk0+sz*k,_plk0[k]);

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
                        catg_largest_p_lk = AVX_Partial_Lk_Inin_Max(_tPij1,plk1,_pmat1plk1,
                                                                    _tPij2,plk2,_pmat2plk2,
                                                                    ns,_plk0);
                        if(catg_largest_p_lk > largest_p_lk) largest_p_lk = catg_largest_p_lk;
                      }
                    else
                      {
                        AVX_Partial_Lk_Inin(_tPij1,plk1,_pmat1plk1,
                                            _tPij2,plk2,_pmat2plk2,
                                            ns,_plk0);
                      }

                    for(k=0;k<nblocks;++k) _mm256_store_pd(plk0+sz*k,_plk0[k]);

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
                
                if(sum_scale[site] >= 1024)
                  {
                    /* plk0 -= ncatgns; */
                    /* plk1 -= (n_v1->tax) ? ns : ncatgns; */
                    /* plk2 -= (n_v2->tax) ? ns : ncatgns; */
                    /* PhyML_Fprintf(stderr,"\n. PARTIAL site: %d plk0: %p [%g %g %g %g] plk1: %p [%g %g %g %g] plk2: %p [%g %g %g %g]", */
                    /*               site, */
                    /*               plk0, */
                    /*               plk0[0], */
                    /*               plk0[1], */
                    /*               plk0[2], */
                    /*               plk0[3], */
                    /*               plk1, */
                    /*               plk1[0], */
                    /*               plk1[1], */
                    /*               plk1[2], */
                    /*               plk1[3], */
                    /*               plk2, */
                    /*               plk2[0], */
                    /*               plk2[1], */
                    /*               plk2[2], */
                    /*               plk2[3] */
                    /*               ); */
                    /* PhyML_Fprintf(stderr,"\n. PARTIAL site: %d d: %d n_v1: %d n_v2: %d",site,d->num,n_v1->num,n_v2->num); */
                    /* PhyML_Fprintf(stderr,"\n. PARTIAL site: %d sum n: %d sum n_v1: %d sum n_v2: %d",site,sum_scale[site],sum_scale_v1_val,sum_scale_v2_val); */
                    
                    /* plk0 += ncatgns; */
                    /* plk1 += (n_v1->tax) ? ns : ncatgns; */
                    /* plk2 += (n_v2->tax) ? ns : ncatgns; */
                    /* Exit("\n"); */
                  }
                
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
            _tPij1 = (__m256d *)init_tPij1;
            _tPij2 = (__m256d *)init_tPij2;

            for(catg=0;catg<ncatg;++catg)
              {                                                          
                if(ambiguity_check_v1 == NO && ambiguity_check_v2 == NO)
                  {
                    AVX_Partial_Lk_Exex(_tPij1,state_v1,
                                        _tPij2,state_v2,
                                        ns,_plk0);
                  }
                else if(ambiguity_check_v1 == YES && ambiguity_check_v2 == NO)
                  {
                    AVX_Partial_Lk_Exin(_tPij2,state_v2,
                                        _tPij1,plk1,_pmat1plk1,
                                        ns,_plk0);
                  }
                else if(ambiguity_check_v1 == NO && ambiguity_check_v2 == YES)
                  {
                    AVX_Partial_Lk_Exin(_tPij1,state_v1,
                                        _tPij2,plk2,_pmat2plk2,
                                        ns,_plk0);
                  }
                else
                  {
                    AVX_Partial_Lk_Inin(_tPij1,plk1,_pmat1plk1,
                                        _tPij2,plk2,_pmat2plk2,
                                        ns,_plk0);
                  }

                for(k=0;k<nblocks;++k) _mm256_store_pd(plk0+sz*k,_plk0[k]);

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

                if(sum_scale[site] >= 1024)
                  {
                  }

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

  if(wght != tree->data->wght)
    PhyML_Copy_Subpatt_Partials(tree,plk0_base,sum_scale,p_lk_loc,ncatg,ns);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void AVX_Partial_Lk_Exex(const __m256d *_tPij1, const int state1, const __m256d *_tPij2, const int state2, const int ns, __m256d *plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  unsigned int i;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      AVX_Partial_Lk_Exex_4(_tPij1,state1,_tPij2,state2,plk0);
      return;
    }
  if(ns == 20)
    {
      AVX_Partial_Lk_Exex_20(_tPij1,state1,_tPij2,state2,plk0);
      return;
    }
#endif

  _tPij1 = _tPij1 + state1 * nblocks;
  _tPij2 = _tPij2 + state2 * nblocks;
  for(i=0;i<nblocks;++i) plk0[i] = _mm256_mul_pd(_tPij1[i],_tPij2[i]);

  /* double *x; */
  /* posix_memalign((void *)&x,BYTE_ALIGN,(size_t)4*sizeof(phydbl)); */
  
  /* _mm256_store_pd(x,plk0[0]); */
  /* for(int i=0;i<4;++i) PhyML_Printf("\n> plk0: %f",x[i]); */

  /*   _mm256_store_pd(x,_tPij1[0]); */
  /* for(int i=0;i<4;++i) PhyML_Printf("\n> Pij1: %f",x[i]); */

  /*   _mm256_store_pd(x,_tPij2[0]); */
  /* for(int i=0;i<4;++i) PhyML_Printf("\n> Pij2: %f",x[i]); */

}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void AVX_Partial_Lk_Exin(const __m256d *_tPij1, const int state1, const __m256d *_tPij2, const phydbl *_plk2, __m256d *_pmat2plk2, const int ns, __m256d *_plk0)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;
  unsigned int i;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      AVX_Partial_Lk_Exin_4(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
      return;
    }
  if(ns == 20)
    {
      AVX_Partial_Lk_Exin_20(_tPij1,state1,_tPij2,_plk2,_pmat2plk2,_plk0);
      return;
    }
#endif
  
  _tPij1 = _tPij1 + state1 * nblocks;
  AVX_Matrix_Vect_Prod(_tPij2,_plk2,ns,_pmat2plk2);
  
  for(i=0;i<nblocks;++i) _plk0[i] = _mm256_mul_pd(_tPij1[i],_pmat2plk2[i]);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void AVX_Partial_Lk_Inin(const __m256d *_tPij1, const phydbl *plk1, __m256d *_pmat1plk1, const __m256d *_tPij2, const phydbl *plk2, __m256d *_pmat2plk2, const int ns, __m256d *_plk0)
{
  unsigned int i;
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;

#if PHYML_OPT_PARTIAL_LK
  if(ns == 4)
    {
      AVX_Partial_Lk_Inin_4(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
      return;
    }
  if(ns == 20)
    {
      AVX_Partial_Lk_Inin_20(_tPij1,plk1,_pmat1plk1,_tPij2,plk2,_pmat2plk2,_plk0);
      return;
    }
#endif

  for(i=0;i<ns;++i) if(plk1[i] > 1.0 || plk1[i] < 1.0 || plk2[i] > 1.0 || plk2[i] < 1.0) break; 

  if(i != ns)
    {      
      AVX_Matrix_Vect_Prod(_tPij1,plk1,ns,_pmat1plk1);
      AVX_Matrix_Vect_Prod(_tPij2,plk2,ns,_pmat2plk2);
      
      for(i=0;i<nblocks;++i) _plk0[i] = _mm256_mul_pd(_pmat1plk1[i],_pmat2plk2[i]);
    }
  else
    {
      for(i=0;i<nblocks;++i) _plk0[i] = _mm256_set1_pd(1.0);      
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void AVX_Matrix_Vect_Prod(const __m256d *_m_transpose, const phydbl *_v, const int ns, __m256d *_u)
{
  unsigned const int sz = (int)BYTE_ALIGN / 8;
  unsigned const int nblocks = ns / sz;  
  unsigned int i,j;  
  __m256d _x;

  _x = _mm256_set1_pd(_v[0]);
  for(j=0;j<nblocks;++j) _u[j] = _mm256_mul_pd(_m_transpose[j],_x);
  _m_transpose = _m_transpose + nblocks;
  
  for(i=1;i<ns;++i)
    {
      _x = _mm256_set1_pd(_v[i]);
      for(j=0;j<nblocks;++j)
        {
#if (defined(__FMA__))
          _u[j] = _mm256_fmadd_pd(_m_transpose[j],_x,_u[j]);
#else
          _u[j] = _mm256_add_pd(_u[j],_mm256_mul_pd(_m_transpose[j],_x));
#endif
        }
      _m_transpose = _m_transpose + nblocks;
    }

/*   for(i=0;i<nblocks;++i) _u[i] = _mm256_setzero_pd(); */
  
/*   for(i=0;i<ns;++i) */
/*     { */
/*       _x = _mm256_set1_pd(_v[i]); */
/*       for(j=0;j<nblocks;++j) */
/*         { */
/* #if (defined(__FMA__)) */
/*           _u[j] = _mm256_fmadd_pd(_m_transpose[j],_x,_u[j]); */
/* #else */
/*           _u[j] = _mm256_add_pd(_u[j],_mm256_mul_pd(_m_transpose[j],_x)); */
/* #endif */
/*         } */
/*       _m_transpose = _m_transpose + nblocks; */
/*     } */

}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

static inline phydbl AVX_Vect_Max(__m256d x)
{
  __m128d vlow;
  __m128d vhigh;
  __m128d vmax;
  __m128d high64;

  vlow = _mm256_castpd256_pd128(x);
  vhigh = _mm256_extractf128_pd(x, 1);
  vmax = _mm_max_pd(vlow, vhigh);
  high64 = _mm_unpackhi_pd(vmax, vmax);
  vmax = _mm_max_sd(vmax, high64);

  return _mm_cvtsd_f64(vmax);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

static inline phydbl AVX_Vects_Max(const __m256d *x, unsigned int nblocks)
{
  phydbl largest_p_lk;
  unsigned int i;

  largest_p_lk = -BIG;
  for(i=0;i<nblocks;++i)
    {
      const phydbl block_max = AVX_Vect_Max(x[i]);
      if(block_max > largest_p_lk) largest_p_lk = block_max;
    }

  return largest_p_lk;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

__m256d AVX_Horizontal_Add(const __m256d x[4])
{
  __m256d y[2],z[2];

  // y[0] = [x00+x01;x10+x11;x02+x03;x12+x13]
  y[0] = _mm256_hadd_pd(x[0], x[1]);
  // y[1] = [x20+x21;x30+x31;x22+x23;x32+x33]
  y[1] = _mm256_hadd_pd(x[2], x[3]);

  // z[0] = [x00+x01;x10+x11;x22+x23;x32+x33]
  /* z[0] = _mm256_blend_pd(y[0],y[1],0b1100); */
  z[0] = _mm256_blend_pd(y[0],y[1],12);
  // z[1] = [x02+x03;x12+x13;x20+x21;x30+x31]
  z[1] = _mm256_permute2f128_pd(y[0],y[1],0x21);

  return(_mm256_add_pd(z[0],z[1]));
}

#endif
