#include "common/BaseCV.hpp"
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/math/constants/constants.hpp>


#define USE_OPTIMIZED_SIN_COS_IN_KMEANS


using namespace std;
using namespace cv;
using namespace boost::math::constants;
using namespace vc_math;


using namespace bt;
using namespace blob;


const cv::Point2f INVALID_POINT = cv::Point2f(-1234.5678, -1234.5678);


// ========== LOCAL HELPER FUNCTIONS ==========
static inline void generateRandomCenter(const vector<Vec2f>& box, float* center, RNG& rng) {
  size_t j, dims = box.size();
  // NOTE: unsure why margin is needed. center[j] = {[0, 1) + [-1/dims, 1/dims)}*range + min
  //float margin = 1.f/dims;
  for( j = 0; j < dims; j++ )
    //center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
    center[j] = ((float)rng)*(box[j][1] - box[j][0]) + box[j][0];
};


static inline float distanceSqrd(const float* a, const float* b, int n, bool simd) {
  if (n == 1) {
    float t = *a - *b;
    return t*t;
  } else {
    int j = 0; float d = 0.f;
  #if CV_SSE
    if( simd )
    {
      float CV_DECL_ALIGNED(16) buf[4];
      __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();

      for( ; j <= n - 8; j += 8 )
      {
        __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
        __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
        d0 = _mm_add_ps(d0, _mm_mul_ps(t0, t0));
        d1 = _mm_add_ps(d1, _mm_mul_ps(t1, t1));
      }
      _mm_store_ps(buf, _mm_add_ps(d0, d1));
      d = buf[0] + buf[1] + buf[2] + buf[3];
    }
    else
  #endif
    {
      for( ; j <= n - 4; j += 4 ) {
        float t0 = a[j] - b[j], t1 = a[j+1] - b[j+1], t2 = a[j+2] - b[j+2], t3 = a[j+3] - b[j+3];
        d += t0*t0 + t1*t1 + t2*t2 + t3*t3;
      }
    }

    for( ; j < n; j++ ) {
      float t = a[j] - b[j];
      d += t*t;
    }
    return d;
  }
};


/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
 */
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
    int K, RNG& rng, int trials)
{
  int i, j, k, dims = _data.cols, N = _data.rows;
  const float* data = _data.ptr<float>(0);
  int step = (int)(_data.step/sizeof(data[0]));
  vector<int> _centers(K);
  int* centers = &_centers[0];
  vector<float> _dist(N*3);
  float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
  double sum0 = 0;
  bool simd = checkHardwareSupport(CV_CPU_SSE);

  centers[0] = (unsigned)rng % N;

  for( i = 0; i < N; i++ )
  {
    dist[i] = distanceSqrd(data + step*i, data + step*centers[0], dims, simd);
    sum0 += dist[i];
  }

  for( k = 1; k < K; k++ )
  {
    double bestSum = DBL_MAX;
    int bestCenter = -1;

    for( j = 0; j < trials; j++ )
    {
      double p = (double)rng*sum0, s = 0;
      for( i = 0; i < N-1; i++ )
        if( (p -= dist[i]) <= 0 )
          break;
      int ci = i;
      for( i = 0; i < N; i++ )
      {
        tdist2[i] = std::min(distanceSqrd(data + step*i, data + step*ci, dims, simd), dist[i]);
        s += tdist2[i];
      }

      if( s < bestSum )
      {
        bestSum = s;
        bestCenter = ci;
        std::swap(tdist, tdist2);
      }
    }
    centers[k] = bestCenter;
    sum0 = bestSum;
    std::swap(dist, tdist);
  }

  for( k = 0; k < K; k++ )
  {
    const float* src = data + step*centers[k];
    float* dst = _out_centers.ptr<float>(k);
    for( j = 0; j < dims; j++ )
      dst[j] = src[j];
  }
};


static inline bool any(const bool* array, unsigned int n) {
  for (unsigned int i = 0; i < n; i++) {
    if (array[i]) return true;
  }
  return false;
};


// WARNING: assuming that cyclical range is [0, 1)!
static inline float circDistanceSqrd(const float* a, const float* b, int n, bool simd) {
  // NOTE: long() casting gives different results compared to both floor() and round()!
  //       this implementation appears to be correct (and is a bit faster than using floor()).
  if (n == 1) {
    float t = *a - *b + 0.5f;
    t = (t > 0) ? t - (long) t - 0.5f : t - (long) t + 0.5f;
    return t*t;
  } else {
    int j = 0; float d = 0.f;
    for (; j <= n - 4; j += 4) {
      float t0 = a[j] - b[j] + 0.5f;
      t0 = (t0 > 0) ? t0 - (long) t0 - 0.5f : t0 - (long) t0 + 0.5f;

      float t1 = a[j+1] - b[j+1] + 0.5f;
      t1 = (t1 > 0) ? t1 - (long) t1 - 0.5f : t1 - (long) t1 + 0.5f;

      float t2 = a[j+2] - b[j+2] + 0.5f;
      t2 = (t2 > 0) ? t2 - (long) t2 - 0.5f : t2 - (long) t2 + 0.5f;

      float t3 = a[j+3] - b[j+3] + 0.5f;
      t3 = (t3 > 0) ? t3 - (long) t3 - 0.5f : t3 - (long) t3 + 0.5f;

      d += t0*t0 + t1*t1 + t2*t2 + t3*t3;
    }
    for (; j < n; j++) {
      float t = a[j] - b[j] + 0.5f;
      t = (t > 0) ? t - (long) t - 0.5f : t - (long) t + 0.5f;
      d += t*t;
    }

    return d;
  }
};


// WARNING: assuming that cyclical range is [0, 1)!
static inline float mixedDistanceSqrd(const float* a, const float* b,
    const bool* isCirc, int n, bool simd) {
  int j = 0; float d = 0.f;
  for (j = 0; j < n; j++) {
    if (isCirc[j]) {
      // NOTE: long() casting gives different results compared to both floor() and round()!
      //       this implementation appears to be correct (and is a bit faster than using floor()).
      float t = a[j] - b[j] + 0.5f;
      t = (t > 0) ? t - (long) t - 0.5f : t - (long) t + 0.5f;
      d += t*t;
    } else {
      float t = a[j] - b[j];
      d += t*t;
    }
  }

  return d;
};


// ========== BaseCV functions ==========
#ifdef CKMEANS_PROFILE
std::vector<unsigned int> BaseCV::ckmeans_iters;
unsigned int BaseCV::ckmeans_term_via_count = 0;
unsigned int BaseCV::ckmeans_term_via_eps = 0;
#endif

#ifdef MIXED_KMEANS_PROFILE
std::vector<unsigned int> BaseCV::mixed_kmeans_iters;
unsigned int BaseCV::mixed_kmeans_term_via_count = 0;
unsigned int BaseCV::mixed_kmeans_term_via_eps = 0;
#endif


#ifdef SOLVE_LINE_RANSAC_PROFILE
std::vector<unsigned int> BaseCV::ransac_iters;
std::vector<double> BaseCV::ransac_first_fit_ratio;
std::vector<double> BaseCV::ransac_second_fit_ratio;
std::vector<double> BaseCV::ransac_first_fit_avg_dist;
std::vector<double> BaseCV::ransac_second_fit_avg_dist;
unsigned int BaseCV::ransac_term_via_iters = 0;
unsigned int BaseCV::ransac_term_via_fit = 0;
#endif


template <class labelType>
double BaseCV::ckmeans(
    const cv::Mat& data,
    labelType K,
    cv::Mat& best_labels,
    const cv::TermCriteria& _criteria,
    int attempts,
    int flags,
    cv::Mat* _centers,
    bool isCircularSpace) {
  // determine properties of input data
  int N = data.rows > 1 ? data.rows : data.cols;
  int dims = (data.rows > 1 ? data.cols : 1)*data.channels();
  int type = data.depth();
  bool simd = checkHardwareSupport(CV_CPU_SSE);
  cv::TermCriteria criteria(_criteria);

  // initialize other internal structures
  // NOTE: if isCircularSpace, centers will accumulate sum(cos(input)) and
  //       centersDual will accumulate sum(sin(input))
  Mat centers(K, dims, type), old_centers(K, dims, type), centersDual(K, dims, type);
  vector<int> counters(K); // Number of entries per cluster
  vector<Vec2f> _box(dims);
  Vec2f* box = &_box[0];

  double best_compactness = DBL_MAX, compactness = 0;
  RNG& rng = theRNG(); // random number generator
  const int SPP_TRIALS = 3; // parameter of kmeans++ centers sampling algorithm
  int a, iter, i, j; // loop indices
  labelType k;
  double max_center_shift;

  // sanity checks
  attempts = std::max(attempts, 1);
  CV_Assert(data.dims <= 2 && type == CV_32F && K > 0);

  if (criteria.type & TermCriteria::EPS) {
    criteria.epsilon = std::max(criteria.epsilon, 0.);
    criteria.maxCount = std::max(criteria.maxCount, 2);
  } else {
    criteria.epsilon = FLT_EPSILON; // minimum positive floating point
  }
  criteria.epsilon *= criteria.epsilon;

  if (criteria.type & TermCriteria::COUNT) {
    criteria.maxCount = std::min(std::max(criteria.maxCount, 1), KMEANS_CRITERIA_MAX_COUNT_CAP);
  } else {
    criteria.maxCount = KMEANS_CRITERIA_MAX_COUNT_CAP;
  }

  // if user specified K = 1 cluster, then compute cluster centers
  // and return immediately after
  if (K == 1) {
    attempts = 1;
    if (criteria.maxCount >= 2) {
      criteria.maxCount = 2; // Set to 2 iterations, so that after initial random centroid seeding, it has a chance to converge
    }
  }

  // pre-compute sines and cosines of angular data once
  Mat dataSin(data.size(), CV_32FC1);
  Mat dataCos(data.size(), CV_32FC1);
  int _cols = data.cols, _rows = data.rows;
  float dataOne, dataRad;
  float* dataSinRow;
  float* dataCosRow;
  float tempSin, tempCos;
  if(data.isContinuous() && dataSin.isContinuous() && dataCos.isContinuous()) {
    _cols *= _rows;
    _rows = 1;
  }
  for(int i = 0; i < _rows; i++) {
    const float* dataRow = data.ptr<float>(i);
    dataSinRow = dataSin.ptr<float>(i);
    dataCosRow = dataCos.ptr<float>(i);
    for(int j = 0; j < _cols; j++) {
#ifdef USE_OPTIMIZED_SIN_COS_IN_KMEANS
      // See: http://lab.polygonal.de/?p=205
      // Justification: input data is discretized between 0 and 179 (cv hue) anyways
      { // Compute sine
        dataOne = dataRow[j];
        if (dataOne > 0.5f) dataOne -= 1.0f;
        dataRad = dataOne * vc_math::two_pi;
        if (dataRad < 0) {
          tempSin = 1.27323954f * dataRad + 0.405284735f * dataRad * dataRad;
          if (tempSin < 0.0f) tempSin = .225f*(tempSin*-tempSin - tempSin) + tempSin;
          else tempSin = .225f*(tempSin*tempSin - tempSin) + tempSin;
        } else {
          tempSin = 1.27323954f * dataRad - 0.405284735f * dataRad * dataRad;
          if (tempSin < 0) tempSin = .225f*(tempSin*-tempSin - tempSin) + tempSin;
          else tempSin = .225f*(tempSin*tempSin - tempSin) + tempSin;
        }
      }
      { // Compute cosine = sine(x + pi/2)
        dataOne = dataRow[j] + 0.25;
        if (dataOne > 0.5f) dataOne -= 1.0f;
        dataRad = dataOne * vc_math::two_pi;
        if (dataRad < 0) {
          tempCos = 1.27323954f * dataRad + 0.405284735f * dataRad * dataRad;
          if (tempCos < 0.0f) tempCos = .225f*(tempCos*-tempCos - tempCos) + tempCos;
          else tempCos = .225f*(tempCos*tempCos - tempCos) + tempCos;
        } else {
          tempCos = 1.27323954f * dataRad - 0.405284735f * dataRad * dataRad;
          if (tempCos < 0) tempCos = .225f*(tempCos*-tempCos - tempCos) + tempCos;
          else tempCos = .225f*(tempCos*tempCos - tempCos) + tempCos;
        }
      }
      dataSinRow[j] = tempSin;
      dataCosRow[j] = tempCos;
#else
      dataRad = dataRow[j] * two_pi;
      dataSinRow[j] = sin(dataRad);
      dataCosRow[j] = cos(dataRad);
#endif
    }
  }

  if (flags & KMEANS_USE_INITIAL_CENTERS) {
    // assume that KMEANS_USE_INITIAL_CENTERS supercedes KMEANS_USE_INITIAL_LABELS
    flags = flags & ~KMEANS_USE_INITIAL_LABELS;

    // copy over user-specified centers into local buffer
    if (_centers != NULL) {
      if (_centers->rows != K || _centers->cols != dims || _centers->depth() != CV_32F) {
        std::cerr << "ckmeans CENTERS SIZE/FORMAT WRONG!, has: " << _centers->rows <<
            ", " << _centers->cols << ", " << _centers->depth() <<
            "; expecting: " << (unsigned short) K << ", " << dims << ", " << CV_32F << std::endl;
      }
      CV_Assert(_centers->rows == K && _centers->cols == dims &&
          _centers->depth() == CV_32F);
      _centers->copyTo(centers);
    }

    // cap circular center values into range [0, 1)
    if (isCircularSpace) {
      MatIterator_<float> it = centers.begin<float>(), it_end = centers.end<float>();
      for(; it != it_end; it++) {
        *it = *it - (long) *it; if (*it < 0) { *it += 1.; } // NOTE: this is equivalent to "*it = *it - floor(*it);", but is slightly faster
      }
    }
  } // else centers will be computed from data in a loop way below

  Mat _labels;
  if (flags & KMEANS_USE_INITIAL_LABELS) { // copy over input labels as initial labels
    CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
        best_labels.cols*best_labels.rows == N &&
        best_labels.isContinuous());
    best_labels.copyTo(_labels);
  } else { // create empty initial labels
//    if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
      if (!( // No need to restrict labelBuffer to be row or column vector, since ckmeans iterates over its entries
        best_labels.cols*best_labels.rows == N &&
        best_labels.isContinuous())) {
      best_labels.create(N, 1, (sizeof(labelType) == 1) ? CV_8U : CV_32S);
    }
    _labels.create(best_labels.size(), best_labels.type());
  }
  labelType* labels = _labels.ptr<labelType>();

  // ensure that initial labels are within the range [0, K)
  if (flags & KMEANS_USE_INITIAL_LABELS) {
    for (i = 0; i < N; i++) {
      CV_Assert((unsigned) labels[i] < (unsigned) K);
    }
  }

  // compute (minimum, maximum) values for all dimensions
  const float* sample = data.ptr<float>(0);
  if (isCircularSpace) {
    for (j = 0; j < dims; j++) {
      box[j] = Vec2f(0.f, 1.f);
    }
  } else {
    // initialize (min, max) to first entry's value, on each dimension
    for (j = 0; j < dims; j++) {
      box[j] = Vec2f(sample[j], sample[j]);
    }
    // update (min, max) by scanning through rest of N - 1 entries
    for (i = 1; i < N; i++) {
      sample = data.ptr<float>(i);
      for (j = 0; j < dims; j++) {
        float v = sample[j];
        box[j][0] = std::min(box[j][0], v);
        box[j][1] = std::max(box[j][1], v);
      }
    }
  }

  // run k-means 'attempts' times using different initial labels
  for (a = 0; a < attempts; a++) {
    max_center_shift = DBL_MAX; // prevent following for-loop from halting before end of second iteration due to epsilon criterion

    // k-means 1-iteration loop
    for (iter = 0; iter < criteria.maxCount && max_center_shift > criteria.epsilon; iter++) {
      if (iter > 0) {
        // Save previous centers (needed to compute max_center_shift)
        swap(centers, old_centers);
      }

      // compute centers for current iteration
      if (iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) &&
          !(flags & KMEANS_USE_INITIAL_CENTERS)) {
        if (flags & KMEANS_PP_CENTERS) { // use kmeans++ initial centers sampling algorithm
          generateCentersPP(data, centers, K, rng, SPP_TRIALS);

          // ensure that circular center values are within [0, 1)
          if (isCircularSpace) {
            MatIterator_<float> it = centers.begin<float>(), it_end = centers.end<float>();
            for(; it != it_end; it++) {
              *it = *it - (long) *it; if (*it < 0) { *it += 1.; } // NOTE: this is equivalent to "*it = *it - floor(*it);", but is slightly faster
            }
          }
        } else { // generate random centers for each of the K clusters
          for (k = 0; k < K; k++) {
            generateRandomCenter(_box, centers.ptr<float>(k), rng);
          }
        }
      } else if (iter > 0 || !(flags & KMEANS_USE_INITIAL_CENTERS)) { // compute centers from existing labels
        centers = Scalar(0);
        if (isCircularSpace) { centersDual = Scalar(0); }
        if (iter > 0) {
          max_center_shift = 0;
        }
        for (k = 0; k < K; k++) {
          counters[k] = 0;
        }

        // compute centers through accumulation of input data
        if (isCircularSpace) { // circular-space centers can be computed as atan2(mean(sin(inputs)), mean(cos(inputs)))
          // sum up contributions from input data
          for (i = 0; i < N; i++) {
            sample = data.ptr<float>(i);
            k = labels[i];
            float* center = centers.ptr<float>(k);
            float* centerDual = centersDual.ptr<float>(k);
            for (j = 0; j < dims; j++) {
              center[j] += dataCos.ptr<float>(i)[j];
              centerDual[j] += dataSin.ptr<float>(i)[j];
            }
            counters[k]++;
          }

          for (k = 0; k < K; k++) {
            // normalize summed center values
            float* center = centers.ptr<float>(k);
            float* centerDual = centersDual.ptr<float>(k);
            if (counters[k] != 0) {
              float scale = 1.f/counters[k];
              for (j = 0; j < dims; j++) {
                center[j] *= scale;
                centerDual[j] *= scale;
                center[j] = atan2(centerDual[j], center[j]) / (vc_math::two_pi);
                if (center[j] < 0.0) { // map range of center[j] to [0, 1)
                  center[j] += 1.0;
                }
              }
            } else { // counters[k] == 0 -> there are no entries in the k-th cluster
              generateRandomCenter(_box, center, rng);
            }

            // update max_center_shift for criteria.epsilon termination criterion
            if (iter > 0) {
              const float* old_center = old_centers.ptr<float>(k);
              double distSqrd = circDistanceSqrd(center, old_center, dims, simd);
              max_center_shift = std::max(max_center_shift, distSqrd);
            }
          }
        } else { // linear space -- original OpenCV code
          // sum up contributions from input data
          if (dims == 1) { // Optimization step
            for (i = 0; i < N; i++) {
              sample = data.ptr<float>(i);
              k = labels[i];
              float* center = centers.ptr<float>(k);
              *center += *sample;
              counters[k]++;
            }
          } else {
            for (i = 0; i < N; i++) {
              sample = data.ptr<float>(i);
              k = labels[i];
              float* center = centers.ptr<float>(k);
              for (j = 0; j <= dims - 4; j += 4) {
                float t0 = center[j] + sample[j];
                float t1 = center[j+1] + sample[j+1];
                float t2 = center[j+2] + sample[j+2];
                float t3 = center[j+3] + sample[j+3];
                center[j] = t0;
                center[j+1] = t1;
                center[j+2] = t2;
                center[j+3] = t3;
              }
              for (; j < dims; j++) {
                center[j] += sample[j];
              }
              counters[k]++;
            }
          }


          for (k = 0; k < K; k++) {
            // normalize summed center values
            float* center = centers.ptr<float>(k);
            if (counters[k] != 0) {
              float scale = 1.f/counters[k];
              for (j = 0; j < dims; j++) {
                center[j] *= scale;
              }
            } else {
              generateRandomCenter(_box, center, rng);
            }

            // update max_center_shift for criteria.epsilon termination criterion
            if (iter > 0) {
              double distSqrd = 0;
              const float* old_center = old_centers.ptr<float>(k);
              for (j = 0; j < dims; j++) {
                double t = center[j] - old_center[j];
                distSqrd += t*t;
              }
              max_center_shift = std::max(max_center_shift, distSqrd);
            }
          }
        } // isCircularSpace
      } // generate initial centers or compute them from data labels
      // (for all cases where flags does not have KMEANS_USE_INITIAL_CENTERS)

      // assign labels based on minimum Euclidean distance criterion
      compactness = 0;
      if (isCircularSpace) { // assign labels (circular space)
        if (dims == 1) { // Optimization step
          for (i = 0; i < N; i++) {
            sample = data.ptr<float>(i);
            labelType k_best = 0;
            double minDistSqrd = DBL_MAX;

            for (k = 0; k < K; k++) {
              const float* center = centers.ptr<float>(k);

              //double distSqrd = circDistanceSqrd(sample, center, dims, simd);
              float mag = *sample - *center + 0.5f;
              mag = (mag > 0) ? mag - (long) mag - 0.5f : mag - (long) mag + 0.5f;
              double distSqrd = mag*mag;

              if (minDistSqrd > distSqrd) {
                minDistSqrd = distSqrd;
                k_best = k;
              }
            }

            compactness += minDistSqrd;
            labels[i] = k_best;
          }
        } else {
          for (i = 0; i < N; i++) {
            sample = data.ptr<float>(i);
            labelType k_best = 0;
            double minDistSqrd = DBL_MAX;

            for (k = 0; k < K; k++) {
              const float* center = centers.ptr<float>(k);
              double distSqrd = circDistanceSqrd(sample, center, dims, simd);

              if (minDistSqrd > distSqrd) {
                minDistSqrd = distSqrd;
                k_best = k;
              }
            }

            compactness += minDistSqrd;
            labels[i] = k_best;
          }
        }
      } else { // assign labels (linear space)
        if (dims == 1) { // Optimization step
          for (i = 0; i < N; i++) {
            sample = data.ptr<float>(i);
            labelType k_best = 0;
            double minDistSqrd = DBL_MAX;

            for (k = 0; k < K; k++) {
              const float* center = centers.ptr<float>(k);

              //double distSqrd = distanceSqrd(sample, center, dims, simd);
              double distSqrd = *sample - *center;
              distSqrd = distSqrd*distSqrd;

              if (minDistSqrd > distSqrd) {
                minDistSqrd = distSqrd;
                k_best = k;
              }
            }

            compactness += minDistSqrd;
            labels[i] = k_best;
          }
        } else {
          for (i = 0; i < N; i++) {
            sample = data.ptr<float>(i);
            labelType k_best = 0;
            double minDistSqrd = DBL_MAX;

            for (k = 0; k < K; k++) {
              const float* center = centers.ptr<float>(k);
              double distSqrd = distanceSqrd(sample, center, dims, simd);

              if (minDistSqrd > distSqrd) {
                minDistSqrd = distSqrd;
                k_best = k;
              }
            }

            compactness += minDistSqrd;
            labels[i] = k_best;
          }
        }
      } // assign labels based on minimum Euclidean distance criterion
    } // k-means 1-iteration loop

#ifdef CKMEANS_PROFILE
    if (max_center_shift <= criteria.epsilon) {
      ckmeans_term_via_eps++;
    } else {
      ckmeans_term_via_count++;
    }
    ckmeans_iters.push_back(iter);

    if (ckmeans_iters.size() > 1) {
      std::cout << "====================" << std::endl;
      std::vector<unsigned int>::iterator itKI = ckmeans_iters.begin();
      std::vector<unsigned int>::iterator itKIEnd = ckmeans_iters.end();
      double ckmeans_sum = 0;
      unsigned int ckmeans_max = numeric_limits<unsigned int>::min();
      unsigned int ckmeans_min = numeric_limits<unsigned int>::max();
      for (; itKI != itKIEnd; itKI++) {
        ckmeans_sum += *itKI;
        if (*itKI > ckmeans_max) { ckmeans_max = *itKI; }
        if (*itKI < ckmeans_min) { ckmeans_min = *itKI; }
      }
      std::cout << "CKMEANS_ITERS MIN-MEAN-MAX: " << ckmeans_min << ", " <<
          ckmeans_sum/ckmeans_iters.size() << ", " << ckmeans_max << std::endl;
      std::cout << "CKMEANS NUM TERMS: " <<
          ckmeans_term_via_count + ckmeans_term_via_eps << ", VIA ITERS: " <<
          ckmeans_term_via_count*100.0/(ckmeans_term_via_count + ckmeans_term_via_eps) <<
          "%, VIA EPS: " <<
          ckmeans_term_via_eps*100.0/(ckmeans_term_via_count + ckmeans_term_via_eps) <<
          "%" << std::endl;
      std::cout << "--------------------" << std::endl << std::flush;
    }
#endif

    // save the k-means run with the most compact result (sum-distance-to-centers)
    if (compactness < best_compactness) {
      best_compactness = compactness;
      if (_centers != NULL) {
        centers.copyTo(*_centers);
      }
      _labels.copyTo(best_labels);
    }
  } // run k-means 'attempts' times using different initial labels

  return best_compactness;
};


template <class labelType>
double BaseCV::mixed_kmeans(
    const cv::Mat& data,
    labelType K,
    cv::Mat& best_labels,
    const cv::TermCriteria& _criteria,
    int attempts,
    int flags,
    cv::Mat* _centers,
    const bool* isCircularSpaces) {
  // determine properties of input data
  int N = data.rows > 1 ? data.rows : data.cols;
  int dims = (data.rows > 1 ? data.cols : 1)*data.channels();
  int type = data.depth();
  bool simd = checkHardwareSupport(CV_CPU_SSE);
  cv::TermCriteria criteria(_criteria);
  bool has_any_circular_spaces = any(isCircularSpaces, dims);

  // initialize other internal structures
  // NOTE: if isCircularSpace, centers will accumulate sum(cos(input)) and
  //       centersDual will accumulate sum(sin(input))
  Mat centers(K, dims, type), old_centers(K, dims, type), centersDual(K, dims, type);
  vector<int> counters(K); // Number of entries per cluster
  vector<Vec2f> _box(dims);
  Vec2f* box = &_box[0];

  double best_compactness = DBL_MAX, compactness = 0;
  RNG& rng = theRNG(); // random number generator
  const int SPP_TRIALS = 3; // parameter of kmeans++ centers sampling algorithm
  int a, iter, i, j; // loop indices
  labelType k;
  double max_center_shift;
  const int CRITERIA_MAX_COUNT_CAP = 1000;
  float currVal;
  float* centersItem;

  // sanity checks
  attempts = std::max(attempts, 1);
  CV_Assert(data.dims <= 2 && type == CV_32F && K > 0);

  if (criteria.type & TermCriteria::EPS) {
    criteria.epsilon = std::max(criteria.epsilon, 0.);
    criteria.maxCount = std::max(criteria.maxCount, 2); // need at least 1 iteration to compute epsilon >change<
  } else {
    criteria.epsilon = FLT_EPSILON; // minimum positive floating point
  }
  criteria.epsilon *= criteria.epsilon;

  if (criteria.type & TermCriteria::COUNT) {
    criteria.maxCount = std::min(std::max(criteria.maxCount, 1), CRITERIA_MAX_COUNT_CAP);
  } else {
    criteria.maxCount = CRITERIA_MAX_COUNT_CAP;
  }

  // if user specified K = 1 cluster, then compute cluster centers
  // and return immediately after
  if (K == 1) {
    attempts = 1;
    if (criteria.maxCount >= 2) {
      criteria.maxCount = 2; // Set to 2 iterations, so that after initial random centroid seeding, it has a chance to converge
    }
  }

  // pre-compute sines and cosines of angular data once
  CV_Assert(data.channels() == 1 && data.cols == dims); // following code assumes single-channel data, where each data variable is a column
  Mat dataSin(data.size(), CV_32FC1);
  Mat dataCos(data.size(), CV_32FC1);
  float dataOne, dataRad;
  float* dataSinRow;
  float* dataCosRow;
  float tempSin, tempCos;

  for(int j = 0; j < data.cols; j++) { // K dimensions
    if (isCircularSpaces[j]) {
      for(int i = 0; i < data.rows; i++) {
        const float* dataRow = data.ptr<float>(i);
        dataSinRow = dataSin.ptr<float>(i);
        dataCosRow = dataCos.ptr<float>(i);
#ifdef USE_OPTIMIZED_SIN_COS_IN_KMEANS
        // See: http://lab.polygonal.de/?p=205
        // Justification: input data is discretized between 0 and 179 (cv hue) anyways
        { // Compute sine
          dataOne = dataRow[j];
          if (dataOne > 0.5f) dataOne -= 1.0f;
          dataRad = dataOne * vc_math::two_pi;
          if (dataRad < 0) {
            tempSin = 1.27323954f * dataRad + 0.405284735f * dataRad * dataRad;
            if (tempSin < 0.0f) tempSin = .225f*(tempSin*-tempSin - tempSin) + tempSin;
            else tempSin = .225f*(tempSin*tempSin - tempSin) + tempSin;
          } else {
            tempSin = 1.27323954f * dataRad - 0.405284735f * dataRad * dataRad;
            if (tempSin < 0) tempSin = .225f*(tempSin*-tempSin - tempSin) + tempSin;
            else tempSin = .225f*(tempSin*tempSin - tempSin) + tempSin;
          }
        }
        { // Compute cosine = sine(x + pi/2)
          dataOne = dataRow[j] + 0.25;
          if (dataOne > 0.5f) dataOne -= 1.0f;
          dataRad = dataOne * vc_math::two_pi;
          if (dataRad < 0) {
            tempCos = 1.27323954f * dataRad + 0.405284735f * dataRad * dataRad;
            if (tempCos < 0.0f) tempCos = .225f*(tempCos*-tempCos - tempCos) + tempCos;
            else tempCos = .225f*(tempCos*tempCos - tempCos) + tempCos;
          } else {
            tempCos = 1.27323954f * dataRad - 0.405284735f * dataRad * dataRad;
            if (tempCos < 0) tempCos = .225f*(tempCos*-tempCos - tempCos) + tempCos;
            else tempCos = .225f*(tempCos*tempCos - tempCos) + tempCos;
          }
        }
        dataSinRow[j] = tempSin;
        dataCosRow[j] = tempCos;
#else
        dataRad = dataRow[j] * vc_math::two_pi;
        dataSinRow[j] = sin(dataRad);
        dataCosRow[j] = cos(dataRad);
#endif
      }
    }
  }

  if (flags & KMEANS_USE_INITIAL_CENTERS) {
    // assume that KMEANS_USE_INITIAL_CENTERS supercedes KMEANS_USE_INITIAL_LABELS
    flags = flags & ~KMEANS_USE_INITIAL_LABELS;

    // copy over user-specified centers into local buffer
    if (_centers != NULL) {
      if (_centers->rows != K || _centers->cols != dims || _centers->depth() != CV_32F) {
        std::cerr << "mixed_kmeans CENTERS SIZE/FORMAT WRONG!, has: " << _centers->rows <<
            ", " << _centers->cols << ", " << _centers->depth() <<
            "; expecting: " << (unsigned short) K << ", " << dims << ", " << CV_32F << std::endl;
      }
      CV_Assert(_centers->rows == K && _centers->cols == dims &&
          _centers->depth() == CV_32F);
      _centers->copyTo(centers);
    }

    // cap circular center values into range [0, 1)
    if (has_any_circular_spaces) {
      for (i = 0; i < centers.rows; i++) {
        centersItem = centers.ptr<float>(i);
        for (j = 0; j < dims; j++) {
          if (isCircularSpaces[j]) {
            currVal = centersItem[j];
            currVal = currVal - (long) currVal;
            if (currVal < 0) { currVal += 1.f; }
            centersItem[j] = currVal;
          }
        }
      }
    }
  } // else centers will be computed from data in a loop way below

  Mat _labels;
  if (flags & KMEANS_USE_INITIAL_LABELS) { // copy over input labels as initial labels
    CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
        best_labels.cols*best_labels.rows == N &&
        best_labels.isContinuous());
    best_labels.copyTo(_labels);
  } else { // create empty initial labels
//    if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
      if (!( // No need to restrict labelBuffer to be row or column vector, since ckmeans iterates over its entries
        best_labels.cols*best_labels.rows == N &&
        best_labels.isContinuous())) {
      best_labels.create(N, 1, (sizeof(labelType) == 1) ? CV_8U : CV_32S);
    }
    _labels.create(best_labels.size(), best_labels.type());
  }
  labelType* labels = _labels.ptr<labelType>();

  // ensure that initial labels are within the range [0, K)
  // NOTE: this isn't that costly in practice... or maybe the compiler
  //       optimized it out in Release+DebugFlags mode
  if (flags & KMEANS_USE_INITIAL_LABELS) {
    for (i = 0; i < N; i++) {
      CV_Assert((unsigned) labels[i] < (unsigned) K);
    }
  }

  // compute (minimum, maximum) values for all dimensions
  const float* sample = data.ptr<float>(0);
  // initialize (min, max) to first entry's value, on each dimension
  for (j = 0; j < dims; j++) {
    if (isCircularSpaces[j]) {
      box[j] = Vec2f(0.f, 1.f);
    } else {
      box[j] = Vec2f(sample[j], sample[j]);
    }
  }
  // update (min, max) by scanning through rest of N - 1 entries
  for (i = 1; i < N; i++) {
    sample = data.ptr<float>(i);
    for (j = 0; j < dims; j++) {
      if (!isCircularSpaces[j]) {
        float v = sample[j];
        box[j][0] = std::min(box[j][0], v);
        box[j][1] = std::max(box[j][1], v);
      }
    }
  }

  // run k-means 'attempts' times using different initial labels
  for (a = 0; a < attempts; a++) {
    max_center_shift = DBL_MAX; // prevent following for-loop from halting before end of second iteration due to epsilon criterion

    // k-means 1-iteration loop
    for (iter = 0; iter < criteria.maxCount && max_center_shift > criteria.epsilon; iter++) {
      if (iter > 0) {
        // Save previous centers (needed to compute max_center_shift)
        swap(centers, old_centers);
      }

      // compute centers for current iteration
      if (iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) &&
          !(flags & KMEANS_USE_INITIAL_CENTERS)) {
        if (flags & KMEANS_PP_CENTERS) { // use kmeans++ initial centers sampling algorithm
          generateCentersPP(data, centers, K, rng, SPP_TRIALS);

          // ensure that circular center values are within [0, 1)
          if (has_any_circular_spaces) {
            for (k = 0; k < K; k++) {
              centersItem = centers.ptr<float>(k);
              for (j = 0; j < dims; j++) {
                if (isCircularSpaces[j]) {
                  currVal = centersItem[j];
                  currVal = currVal - (long) currVal;
                  if (currVal < 0) { currVal += 1.f; }
                  centersItem[j] = currVal;
                }
              }
            }
          }
        } else { // generate random centers for each of the K clusters
          for (k = 0; k < K; k++) {
            generateRandomCenter(_box, centers.ptr<float>(k), rng);
          }
        }
      } else if (iter > 0 || !(flags & KMEANS_USE_INITIAL_CENTERS)) { // compute centers from existing labels
        centers = Scalar(0);
        if (has_any_circular_spaces) {
          centersDual = Scalar(0);
        }
        if (iter > 0) {
          max_center_shift = 0;
        }
        for (k = 0; k < K; k++) {
          counters[k] = 0;
        }

        // compute centers through accumulation of input data
        // NOTE: circular-space centers can be computed as
        //       atan2(mean(sin(inputs)), mean(cos(inputs)))
        // sum up contributions from input data
        for (i = 0; i < N; i++) {
          sample = data.ptr<float>(i);
          k = labels[i];
          float* center = centers.ptr<float>(k);
          float* centerDual = centersDual.ptr<float>(k);
          for (j = 0; j < dims; j++) {
            if (isCircularSpaces[j]) {
              // NOTE: this is the non-pre-computed sin/cos code
              //float sampleRad = sample[j]*two_pi;
              //center[j] += cos(sampleRad);
              //centerDual[j] += sin(sampleRad);

              center[j] += dataCos.ptr<float>(i)[j];
              centerDual[j] += dataSin.ptr<float>(i)[j];
            } else {
              center[j] += sample[j];
            }
          }
          counters[k]++;
        }

        for (k = 0; k < K; k++) {
          // normalize summed center values
          float* center = centers.ptr<float>(k);
          float* centerDual = centersDual.ptr<float>(k);
          if (counters[k] != 0) {
            float scale = 1.f/counters[k];
            for (j = 0; j < dims; j++) {
              if (isCircularSpaces[j]) {
                center[j] *= scale;
                centerDual[j] *= scale;
                center[j] = atan2(centerDual[j], center[j]) / vc_math::two_pi;
                if (center[j] < 0.0) { // map range of center[j] to [0, 1)
                  center[j] += 1.0;
                }
              } else { // Not circular space
                center[j] *= scale;
              }
            }
          } else { // counters[k] == 0 -> there are no entries in the k-th cluster
            generateRandomCenter(_box, center, rng);
          }

          // update max_center_shift for criteria.epsilon termination criterion
          if (iter > 0) {
            double distSqrd = 0;
            const float* old_center = old_centers.ptr<float>(k);
            for (j = 0; j < dims; j++) {
              if (isCircularSpaces[j]) {
                // compute angular-[0,1) distance between the two centers
                double t = center[j] - old_center[j] + 0.5;
                t = (t > 0) ? t - (long) t - 0.5f : t - (long) t + 0.5f;
                distSqrd += t*t;
              } else { // Not circular space
                double t = center[j] - old_center[j];
                distSqrd += t*t;
              }
            }
            max_center_shift = std::max(max_center_shift, distSqrd);
          }
        } // normalize summed center values
      } // generate initial centers or compute them from data labels
      // (for all cases where flags does not have KMEANS_USE_INITIAL_CENTERS)

      // assign labels based on minimum Euclidean distance criterion
      compactness = 0;

      if (dims == 2 && isCircularSpaces[0] && !isCircularSpaces[1]) { // Highly value-specific optimization
        float mag;
        double distSqrd;

        for (i = 0; i < N; i++) {
          sample = data.ptr<float>(i);
          labelType k_best = 0;
          double minDistSqrd = DBL_MAX;

          for (k = 0; k < K; k++) {
            const float* center = centers.ptr<float>(k);

            //double distSqrd = mixedDistanceSqrd(sample, center, isCircularSpaces, dims, simd);
            mag = sample[0] - center[0] + 0.5f;
            mag = (mag > 0) ? mag - (long) mag - 0.5f : mag - (long) mag + 0.5f;
            distSqrd = mag*mag;
            mag = sample[1] - center[1];
            distSqrd += mag*mag;

            if (minDistSqrd > distSqrd) {
              minDistSqrd = distSqrd;
              k_best = k;
            }
          }

          compactness += minDistSqrd;
          labels[i] = k_best;
        } // assign labels based on minimum Euclidean distance criterion
      } else {
        for (i = 0; i < N; i++) {
          sample = data.ptr<float>(i);
          labelType k_best = 0;
          double minDistSqrd = DBL_MAX;

          for (k = 0; k < K; k++) {
            const float* center = centers.ptr<float>(k);
            double distSqrd = mixedDistanceSqrd(sample, center, isCircularSpaces, dims, simd);

            if (minDistSqrd > distSqrd) {
              minDistSqrd = distSqrd;
              k_best = k;
            }
          }

          compactness += minDistSqrd;
          labels[i] = k_best;
        } // assign labels based on minimum Euclidean distance criterion
      }
    } // k-means 1-iteration loop

#ifdef MIXED_KMEANS_PROFILE
    if (max_center_shift <= criteria.epsilon) {
      mixed_kmeans_term_via_eps++;
    } else {
      mixed_kmeans_term_via_count++;
    }
    mixed_kmeans_iters.push_back(iter);

    if (mixed_kmeans_iters.size() > 1) {
      std::cout << "====================" << std::endl;
      std::vector<unsigned int>::iterator itKI = mixed_kmeans_iters.begin();
      std::vector<unsigned int>::iterator itKIEnd = mixed_kmeans_iters.end();
      double mixed_kmeans_sum = 0;
      unsigned int mixed_kmeans_max = numeric_limits<unsigned int>::min();
      unsigned int mixed_kmeans_min = numeric_limits<unsigned int>::max();
      for (; itKI != itKIEnd; itKI++) {
        mixed_kmeans_sum += *itKI;
        if (*itKI > mixed_kmeans_max) { mixed_kmeans_max = *itKI; }
        if (*itKI < mixed_kmeans_min) { mixed_kmeans_min = *itKI; }
      }
      std::cout << "MIXED_KMEANS_ITERS MIN-MEAN-MAX: " << mixed_kmeans_min << ", " <<
          mixed_kmeans_sum/mixed_kmeans_iters.size() << ", " << mixed_kmeans_max << std::endl;
      std::cout << "MIXED_KMEANS NUM TERMS: " <<
          mixed_kmeans_term_via_count + mixed_kmeans_term_via_eps << ", VIA ITERS: " <<
          mixed_kmeans_term_via_count*100.0/(mixed_kmeans_term_via_count + mixed_kmeans_term_via_eps) <<
          "%, VIA EPS: " <<
          mixed_kmeans_term_via_eps*100.0/(mixed_kmeans_term_via_count + mixed_kmeans_term_via_eps) <<
          "%" << std::endl;
      std::cout << "--------------------" << std::endl << std::flush;
    }
#endif

    // save the k-means run with the most compact result (sum-distance-to-centers)
    if (compactness < best_compactness) {
      best_compactness = compactness;
      if (_centers != NULL) {
        centers.copyTo(*_centers);
      }
      _labels.copyTo(best_labels);
    }
  } // run k-means 'attempts' times using different initial labels

  return best_compactness;
};


void BaseCV::segmentHue(
    const cv::Mat& bgrImage,
    unsigned char K,
    const cv::TermCriteria& termCrit,
    cv::Mat& labelBuffer,
    cv::Mat& centersBuffer,
    cv::Mat& dataBuffer,
    cv::Mat& hsvImage,
    cv::Mat& hueImage) {
  // DEPRECATED: Blur the image using a Gaussian
  /*
  cv::Size ksize(11, 11);
  double sigma = 2.0;
  cv::Mat bgrBlurredBuffer;
  bgrBlurredBuffer.create(bgrImage.size(), CV_8UC3);
  cv::GaussianBlur(bgrImage, bgrBlurredBuffer, ksize, sigma);
  cvtColor(bgrBlurredBuffer, hsvBuffer, CV_RGB2HSV);
  */

  // Obtain hue vector
  cvtColor(bgrImage, hsvImage, CV_BGR2HSV); // WARNING: loss of precision -- CV_BGR2HSV's conversion discretizes hue into [0, 180]
  int hsv2hueID[] = {0, 0};
  hueImage.create(hsvImage.size(), CV_8UC1); // Initiate hueBuffer (as stipulated by doc. of mixChannels)
  mixChannels(&hsvImage, 1, &hueImage, 1, hsv2hueID, 1); // NOTE: hue values are stored as unsigned chars, in range [0, 180]

  // Compute labels using the core of the K-Means algorithm
  hueImage.convertTo(dataBuffer, CV_32F, 1/180.f, 0);
  dataBuffer = dataBuffer.reshape(0, dataBuffer.rows * dataBuffer.cols);
  labelBuffer.create(hueImage.size(), CV_8UC1);
  ckmeans<unsigned char>(dataBuffer, K, labelBuffer, termCrit,
      KMEANS_ATTEMPTS, KMEANS_FLAGS, &centersBuffer, true);
};


void BaseCV::segmentHueVal(
    const cv::Mat& bgrImage,
    unsigned char K,
    const cv::TermCriteria& termCrit,
    cv::Mat& labelBuffer,
    cv::Mat& centersBuffer,
    cv::Mat& dataBuffer,
    cv::Mat& hsvImage) {
  // Obtain h/v vector by capping sat value
  cvtColor(bgrImage, hsvImage, CV_BGR2HSV); // WARNING: loss of precision -- CV_BGR2HSV's conversion discretizes hue into [0, 180]
  dataBuffer.create(hsvImage.rows * hsvImage.cols, 2, CV_32F);
  MatIterator_<Vec3b> itHSV = hsvImage.begin<Vec3b>();
  MatIterator_<Vec3b> itHSVEnd = hsvImage.end<Vec3b>();
  float* currHVEntry;
  const unsigned char SEGMENT_HUE_VAL_SAT_THRESHOLD_BYTE = SEGMENT_HUE_VAL_SAT_THRESHOLD_RATIO*255;
  for (unsigned int i = 0; itHSV != itHSVEnd; itHSV++, i++) {
    currHVEntry = dataBuffer.ptr<float>(i);
    if ((*itHSV)[1] <= SEGMENT_HUE_VAL_SAT_THRESHOLD_BYTE) {
      currHVEntry[0] = SEGMENT_HUE_VAL_DEFAULT_HUE;
      currHVEntry[1] = float((*itHSV)[2])/255.0;
    } else {
      currHVEntry[0] = float((*itHSV)[0])/180.0;
      currHVEntry[1] = SEGMENT_HUE_VAL_DEFAULT_VAL;
    }
  }

  // Compute labels using the core of the K-Means algorithm
  labelBuffer.create(hsvImage.size(), CV_8UC1);
  bool isCircularSpaces[] = {true, false};
  mixed_kmeans<unsigned char>(dataBuffer, K, labelBuffer, termCrit,
      KMEANS_ATTEMPTS, KMEANS_FLAGS, &centersBuffer, isCircularSpaces);
};


void BaseCV::segmentGrayscale(
    const cv::Mat& bgrImage,
    unsigned char K,
    const cv::TermCriteria& termCrit,
    cv::Mat& labelBuffer,
    cv::Mat& centersBuffer,
    cv::Mat& dataBuffer,
    cv::Mat& grayImage) {
  // Obtain grayscale vector
  cvtColor(bgrImage, grayImage, CV_BGR2GRAY);

  // Compute labels using the core of the kmeans algorithm
  grayImage.convertTo(dataBuffer, CV_32F, 1/255.f, 0);
  dataBuffer = dataBuffer.reshape(0, dataBuffer.rows * dataBuffer.cols);
  labelBuffer.create(grayImage.size(), CV_8UC1);
  ckmeans<unsigned char>(dataBuffer, K, labelBuffer, termCrit,
      KMEANS_ATTEMPTS, KMEANS_FLAGS, &centersBuffer, false);
};


bool BaseCV::estimateHueCenters(
    const cv::Mat& hueImage,
    cv::Mat& centersBuffer,
    unsigned int kmeansK,
    double splitHeading) {
  double cx = hueImage.cols/2.0 - 0.5;
  double cy = hueImage.rows/2.0 - 0.5;
  int currX, currY;
  double currHue, currHeading, headingDiff;
  double cluster1sinSum = 0, cluster1cosSum = 0;
  double cluster2sinSum = 0, cluster2cosSum = 0;
  double cluster1Count = 0, cluster2Count = 0;
  double cluster1Heading, cluster2Heading;

#ifdef ESTIMATE_HUE_CENTERS_SHOW_IMAGE
  cv::Mat hueLabels = Mat::zeros(hueImage.size(), CV_8UC1);
#endif

  splitHeading += 90.0; // Split the image by computing the distance to the tangential image
  for (currY = 0; currY < hueImage.rows; currY++) {
    const unsigned char* hueImageRow = hueImage.ptr<unsigned char>(currY);
#ifdef ESTIMATE_HUE_CENTERS_SHOW_IMAGE
    unsigned char* hueLabelsRow = hueLabels.ptr<unsigned char>(currY);
#endif
    for (currX = 0; currX < hueImage.cols; currX++) {
      currHeading = atan2(currX - cx, cy - currY) * radian;
      headingDiff = angularDist(currHeading, splitHeading);
      currHue = hueImageRow[currX] * degree * 2; // NOTE: *2 since OpenCV maps hue to (char) [0, 180) range
      if (headingDiff <= 90.0) {
        cluster1Count += 1;
        cluster1sinSum += sin(currHue);
        cluster1cosSum += cos(currHue);
      } else {
        cluster2Count += 1;
        cluster2sinSum += sin(currHue);
        cluster2cosSum += cos(currHue);
#ifdef ESTIMATE_HUE_CENTERS_SHOW_IMAGE
        hueLabelsRow[currX] = 255;
#endif
      }
    }
  }

  centersBuffer.create(kmeansK, 1, CV_32F);
  if (cluster1Count > 0 && cluster2Count > 0) {
    cluster1Heading = atan2(cluster1sinSum / cluster1Count,
        cluster1cosSum / cluster1Count) / vc_math::two_pi;
    if (cluster1Heading < 0.0) { cluster1Heading += 1.0; }
    cluster2Heading = atan2(cluster2sinSum / cluster2Count,
        cluster2cosSum / cluster2Count) / vc_math::two_pi;
    if (cluster2Heading < 0.0) { cluster2Heading += 1.0; }

    centersBuffer.at<float>(0, 0) = cluster1Heading;
    centersBuffer.at<float>(1, 0) = cluster2Heading;

#ifdef ESTIMATE_HUE_CENTERS_SHOW_IMAGE
    cv::Mat hsvCenters(2, 1, CV_8UC3), bgrCenters;
    hsvCenters.at<Vec3b>(0, 0) = \
        cv::Vec3b(floor(cluster1Heading*180.f),
        DEFAULT_HSV_SATURATION, DEFAULT_HSV_VALUE);
    hsvCenters.at<Vec3b>(1, 0) = \
        cv::Vec3b(floor(cluster2Heading*180.f),
        DEFAULT_HSV_SATURATION, DEFAULT_HSV_VALUE);
    cvtColor(hsvCenters, bgrCenters, CV_HSV2BGR);
    Vec3b* bgrCentersPtr = bgrCenters.ptr<Vec3b>();

    cv::Mat bgrImage = cv::Mat(hueImage.size(), CV_8UC3, Scalar(bgrCentersPtr[0]));
    bgrImage.setTo(Scalar(bgrCentersPtr[1]), hueLabels);
    imshow("estimateHueCenters() result", bgrImage);
#endif

    return true;
  }

  return false;
};


unsigned char BaseCV::binarizeLabels(
    const cv::Mat& labelBuffer,
    const cv::Mat& centersBuffer,
    cv::Mat& binLabelBuffer,
    int clusteringType) {
  unsigned char bestLabelID = 0;
  unsigned char currLabelID;
  double bestHueDist, bestValDist, currHueDist, currValDist;
  MatConstIterator_<float> itCenters, itCentersEnd;

  switch (clusteringType) {
  case HUE_VAL_CLUSTERING:
    bestHueDist = angularDist(centersBuffer.at<float>(0, 0),
        BINARIZE_LABEL_PREFERRED_HUE, 1.f);
    bestValDist = fabs(centersBuffer.at<float>(0, 1) -
        BINARIZE_LABEL_PREFERRED_VAL);

    for (currLabelID = 1; currLabelID < centersBuffer.rows; currLabelID++) {
      currHueDist = angularDist(centersBuffer.at<float>(currLabelID, 0),
          BINARIZE_LABEL_PREFERRED_HUE, 1.f);
      currValDist = fabs(centersBuffer.at<float>(currLabelID, 1) -
          BINARIZE_LABEL_PREFERRED_VAL);
      currHueDist = currHueDist + currValDist;
      if (currHueDist < bestHueDist) {
        bestValDist = currValDist;
        bestHueDist = currHueDist;
      }
    }
    break;
  case HUE_CLUSTERING:
    itCenters = centersBuffer.begin<float>();
    itCentersEnd = centersBuffer.end<float>();
    bestHueDist = angularDist(*itCenters, BINARIZE_LABEL_PREFERRED_HUE, 1.f);
    currLabelID = 1;
    for (itCenters++; itCenters != itCentersEnd; itCenters++, currLabelID++) {
      currHueDist = angularDist(*itCenters, BINARIZE_LABEL_PREFERRED_HUE, 1.f);
      if (currHueDist < bestHueDist) {
        bestHueDist = currHueDist;
        bestLabelID = currLabelID;
      }
    }
    break;
  case GRAYSCALE_CLUSTERING:
    itCenters = centersBuffer.begin<float>();
    itCentersEnd = centersBuffer.end<float>();
    bestValDist = fabs(*itCenters - BINARIZE_LABEL_PREFERRED_GRAYSCALE);
    currLabelID = 1;
    for (itCenters++; itCenters != itCentersEnd; itCenters++, currLabelID++) {
      currValDist = fabs(*itCenters - BINARIZE_LABEL_PREFERRED_GRAYSCALE);
      if (currValDist < bestValDist) {
        bestValDist = currValDist;
        bestLabelID = currLabelID;
      }
    }
    break;
  default:
    // NOTE: already set bestLabelID = 0 by default
    break;
  }

  binLabelBuffer = (labelBuffer != bestLabelID)/255; // Returns CV_8UC1 type cv::Mat
  return bestLabelID;
};


void BaseCV::applyModeFilter(
    cv::Mat& binLabelBuffer,
    cv::Mat& binLabelMask,
    double boxWidthRatio) {
#ifdef MODE_FILTER_SHOW_FILTER_DIFF
  cv::Mat origBinLabelBuffer;
  binLabelBuffer.convertTo(origBinLabelBuffer, CV_8UC1);
#endif

  unsigned int imHeight = binLabelBuffer.rows, imWidth = binLabelBuffer.cols;

  unsigned int boxWidth = floor(std::min(imHeight, imWidth)*boxWidthRatio);
  boxWidth = std::max((unsigned int) 1, std::min(boxWidth, std::min(imHeight, imWidth)));

  // Perform box filter on center range of integral image (i.e. 4 matrix operations)
  cv::Mat integralBuffer, boxCenterBuffer, modeLabelBuffer;
  integral(binLabelBuffer, integralBuffer); // WARNING: integral image will have an extra top-row and left-column of zeros
  unsigned int boxHalfWidth = floor(boxWidth/2); // WARNING: take note of the floor(.)
  boxCenterBuffer =
      integralBuffer(Range(0, imHeight - boxWidth + 1), Range(0, imWidth - boxWidth + 1)) +
      integralBuffer(Range(boxWidth + 0, imHeight + 1), Range(boxWidth + 0, imWidth + 1)) -
      integralBuffer(Range(0, imHeight - boxWidth + 1), Range(boxWidth + 0, imWidth + 1)) -
      integralBuffer(Range(boxWidth + 0, imHeight + 1), Range(0, imWidth - boxWidth + 1));
#ifdef MODE_FILTER_ZERO_BORDERS
  binLabelBuffer = Scalar(0);
#endif
  binLabelBuffer(
      Range(boxHalfWidth, boxHalfWidth + imHeight - boxWidth + 1),
      Range(boxHalfWidth, boxHalfWidth + imWidth - boxWidth + 1)) =
      (boxCenterBuffer > floor(boxWidth*boxWidth/2)) / 255;

  // Compute selection mask
  binLabelMask = Mat::zeros(binLabelBuffer.size(), CV_8UC1);
  binLabelMask(
      Range(boxHalfWidth + 1, boxHalfWidth + imHeight - boxWidth),
      Range(boxHalfWidth + 1, boxHalfWidth + imWidth - boxWidth)) = Scalar(255);

#ifdef MODE_FILTER_BLUR_BORDERS
  // Perform reduced box filter on image borders
  // WARNING: boxWidth and boxHalfWidth are re-defined in this loop
  unsigned int defaultBoxHalfWidth = boxHalfWidth;
  for (boxHalfWidth = 1; boxHalfWidth <= defaultBoxHalfWidth; boxHalfWidth++) {
    boxWidth = boxHalfWidth*2 + 1;

    // Process top row
    boxCenterBuffer = \
        integralBuffer(Range(0, 1), Range(0, imWidth - boxWidth + 1)) + \
        integralBuffer(Range(boxWidth, boxWidth + 1), Range(boxWidth, imWidth + 1)) - \
        integralBuffer(Range(0, 1), Range(boxWidth, imWidth + 1)) - \
        integralBuffer(Range(boxWidth, boxWidth + 1), Range(0, imWidth - boxWidth + 1));
    binLabelBuffer( \
        Range(boxHalfWidth, boxHalfWidth + 1), \
        Range(boxHalfWidth, boxHalfWidth + imWidth - boxWidth + 1)) = \
        (boxCenterBuffer > floor(boxWidth*boxWidth/2)) / 255;

    // Process bottom row
    boxCenterBuffer = \
        integralBuffer(Range(imHeight - boxWidth, imHeight - boxWidth + 1), Range(0, imWidth - boxWidth + 1)) + \
        integralBuffer(Range(imHeight, imHeight + 1), Range(boxWidth + 0, imWidth + 1)) - \
        integralBuffer(Range(imHeight - boxWidth, imHeight - boxWidth + 1), Range(boxWidth + 0, imWidth + 1)) - \
        integralBuffer(Range(imHeight, imHeight + 1), Range(0, imWidth - boxWidth + 1));
    binLabelBuffer( \
        Range(boxHalfWidth + imHeight - boxWidth, boxHalfWidth + imHeight - boxWidth + 1), \
        Range(boxHalfWidth, boxHalfWidth + imWidth - boxWidth + 1)) = \
        (boxCenterBuffer > floor(boxWidth*boxWidth/2)) / 255;

    // Process left column
    boxCenterBuffer = \
        integralBuffer(Range(0, imHeight - boxWidth + 1), Range(0, 1)) + \
        integralBuffer(Range(boxWidth + 0, imHeight + 1), Range(boxWidth, boxWidth + 1)) - \
        integralBuffer(Range(0, imHeight - boxWidth + 1), Range(boxWidth, boxWidth + 1)) - \
        integralBuffer(Range(boxWidth + 0, imHeight + 1), Range(0, 1));
    binLabelBuffer( \
        Range(boxHalfWidth, boxHalfWidth + imHeight - boxWidth + 1), \
        Range(boxHalfWidth, boxHalfWidth + 1)) = \
        (boxCenterBuffer > floor(boxWidth*boxWidth/2)) / 255;

    // Process right column
    boxCenterBuffer = \
        integralBuffer(Range(0, imHeight - boxWidth + 1), Range(imWidth - boxWidth, imWidth - boxWidth + 1)) + \
        integralBuffer(Range(boxWidth + 0, imHeight + 1), Range(imWidth, imWidth + 1)) - \
        integralBuffer(Range(0, imHeight - boxWidth + 1), Range(imWidth, imWidth + 1)) - \
        integralBuffer(Range(boxWidth + 0, imHeight + 1), Range(imWidth - boxWidth, imWidth - boxWidth + 1));
    binLabelBuffer( \
        Range(boxHalfWidth, boxHalfWidth + imHeight - boxWidth + 1), \
        Range(boxHalfWidth + imWidth - boxWidth, boxHalfWidth + imWidth - boxWidth + 1)) = \
        (boxCenterBuffer > floor(boxWidth*boxWidth/2)) / 255;
  }
#endif

#ifdef MODE_FILTER_SHOW_FILTER_DIFF
  cv::Mat diffBuffer = (origBinLabelBuffer != binLabelBuffer);
  imshow("Box Filter Difference Image", diffBuffer);
#endif
};


void BaseCV::computeDisjointSets(
    const cv::Mat& binLabelImage,
    cv::Mat& imageIDs,
    std::vector<unsigned int>& parentIDs,
    std::vector<unsigned char>& setLabels,
    bool eightConnected) {
  int currY, currX;
  unsigned int topID, leftID, topLeftID, newID, parentID;
  const unsigned char* binLabelRow = NULL;
  const unsigned char* prevBinLabelRow = NULL;
  unsigned int* imageIDRow = NULL;
  unsigned int* prevImageIDRow = NULL;
  bool eightConnMerged = false;

  // Ensure that there are enough labels for all possible pixels in the image
  CV_Assert(binLabelImage.rows * binLabelImage.cols < pow(2, sizeof(unsigned int)*8));

  // Create imageIDs buffer if necessary
  imageIDs.create(binLabelImage.size(), CV_32SC1);
  // NOTE: even though we are specifying type == signed int, we can still be
  //       self-consistent and use it as unsigned int

  // Computes disjoint set IDs for pixels
  //
  // WARNING: this may create N-step parent invariance for arbitrarily large N, e.g.
  //
  // MATRIX: 5 rows x 9 cols
  // 0 1 0 1 0 1 0 1 0
  // 0 1 0 1 0 1 0 0 0
  // 0 1 0 1 0 0 0 0 0
  // 0 1 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0
  //
  // IMAGE IDs: 5 rows x 9 cols
  // 0 1 2 3 4 5 6 7 8
  // 0 1 2 3 4 5 6 6 6
  // 0 1 2 3 4 4 4 4 4
  // 0 1 2 2 2 2 2 2 2
  // 0 0 0 0 0 0 0 0 0
  //
  //
  // Parent of 0: 0
  // Parent of 1: 1
  // Parent of 2: 0
  // Parent of 3: 3
  // Parent of 4: 2
  // Parent of 5: 5
  // Parent of 6: 4
  // Parent of 7: 7
  // Parent of 8: 6
  //
  // ... nevertheless, all ancestors of each pixel/parent ID will have the invariance
  // that their parent ID number is smaller than their current ID. Thus we can
  // re-establish 1-step parent invariance by setting the parent ID of each
  // parent to be their 2-step ancestor, starting with the smallest-numbered parent IDs.
  // Subsequently, we can re-establish 0-step parent invariance by updating each
  // image ID to their parent ID.
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    prevBinLabelRow = binLabelRow;
    prevImageIDRow = imageIDRow;
    binLabelRow = binLabelImage.ptr<unsigned char>(currY);
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      if (currX == 0) {
        if (currY == 0) { // Case 0 = Case 0-NEW: (TOP-LEFT CORNER) First pixel, so no neighbors to check with
          parentIDs.clear(); // Necessary redundancy, since buffer is external
          setLabels.clear(); // Necessary redundancy, since buffer is external
          imageIDRow[currX] = 0; // Assign pixel to set ID == 0
          parentIDs.push_back(0); // parent ID = self ID
          setLabels.push_back(binLabelRow[currX]);
        } else { // Case 1: (LEFT COLUMN) Check with top neighbor
          if (binLabelRow[currX] == prevBinLabelRow[currX]) { // Case 1-MERGE
            topID = parentIDs[parentIDs[prevImageIDRow[currX]]]; // NOTE: need 2-step parent, otherwise top-right/bottom-left 8-connected neighbours may not merge properly
            imageIDRow[currX] = topID;
          } else { // Case 1-NEW
            newID = parentIDs.size(); // assign new ID to current pixel
            imageIDRow[currX] = newID;
            parentIDs.push_back(newID);
            setLabels.push_back(binLabelRow[currX]);
          }
        }
      } else if (currY == 0) { // Case 2: (TOP ROW) Check with left neighbor
        if (binLabelRow[currX] == binLabelRow[currX - 1]) { // Case 2-MERGE
          leftID = parentIDs[parentIDs[imageIDRow[currX - 1]]]; // NOTE: need 2-step parent, otherwise top-right/bottom-left 8-connected neighbours may not merge properly
          imageIDRow[currX] = leftID;
        } else { // Case 2-NEW
          newID = parentIDs.size(); // assign new ID to current pixel
          imageIDRow[currX] = newID;
          parentIDs.push_back(newID);
          setLabels.push_back(binLabelRow[currX]);
        }
      } else { // Case 3: (NON-LEFT COLUMN AND NON-TOP ROW) Check with top and left neighbors
        topID = parentIDs[parentIDs[prevImageIDRow[currX]]]; // NOTE: need 2-step parent, otherwise top-right/bottom-left 8-connected neighbours may not merge properly
        leftID = parentIDs[parentIDs[imageIDRow[currX - 1]]]; // NOTE: need 2-step parent, otherwise top-right/bottom-left 8-connected neighbours may not merge properly
        if (binLabelRow[currX] == prevBinLabelRow[currX]) {
          if (binLabelRow[currX] == binLabelRow[currX - 1]) { // Case 3-MERGE-ALL: Top pixel, left pixel, and current pixel all have same label, so merge set
            // Merge top and left IDs, favoring the smaller ID
            // WARNING: this merging may create a new 1-step parent chain locally, which in the worst case
            //          may create N-step parent invariance globally with for arbitrarily large N
            if (leftID < topID) {
              parentIDs[topID] = leftID; // <- creates an additional 1-step parent chain locally
              imageIDRow[currX] = leftID;
            } else {
              parentIDs[leftID] = topID; // <- creates an additional 1-step parent chain locally
              imageIDRow[currX] = topID;
            }
          } else { // Case 3-MERGE-TOP: Only top pixel and current pixel have same label
            imageIDRow[currX] = topID;
          }
        } else { // binLabelRow[currX] != prevBinLabelRow[currX]
          if (binLabelRow[currX] == binLabelRow[currX - 1]) { // Case 3-MERGE-LEFT: Only left pixel and current pixel have same label
            imageIDRow[currX] = leftID;
          } else { // Case 3-MERGE-NEW: Top and left pixel have the same label, but current pixel has different label, so (in 8-conn mode merge top and left pixels, then) assign new ID to current pixel
            eightConnMerged = false;
            if (eightConnected) {
              if (binLabelRow[currX - 1] == prevBinLabelRow[currX]) { // Redundancy check
                // Merge top and left IDs, favoring the smaller ID
                // WARNING: this merging may create a new 1-step parent chain locally, which in the worst case
                //          may create N-step parent invariance globally with for arbitrarily large N
                if (leftID < topID) {
                  parentIDs[topID] = leftID; // <- creates an additional 1-step parent chain locally
                } else {
                  parentIDs[leftID] = topID; // <- creates an additional 1-step parent chain locally
                }
              } // Else binary label isn't binary!
              else {
                cerr << "INTERNAL ERROR: NON-BINARY LABELS FOUND" << endl;
                cerr << currY-1 << ", " << currX << ": " << short(prevBinLabelRow[currX]) << endl;
                cerr << currY << ", " << currX-1 << ": " << short(binLabelRow[currX - 1]) << endl;
                cerr << currY << ", " << currX << ": " << short(binLabelRow[currX]) << endl;
                CV_Assert(false);
              }

              // Also merge up-left pixel to current pixel if they share the same label
              if (binLabelRow[currX] == prevBinLabelRow[currX - 1]) {
                topLeftID = parentIDs[parentIDs[prevImageIDRow[currX - 1]]];
                imageIDRow[currX] = topLeftID;
                eightConnMerged = true;
              }
            }

            // Assign new set ID to current pixel
            if (!eightConnMerged) { // either did not merge with top-left neighbor, or is in 4-connected mode
              newID = parentIDs.size();
              imageIDRow[currX] = newID;
              parentIDs.push_back(newID);
              setLabels.push_back(binLabelRow[currX]);
            }
          }
        }
      } // 4 cases
    } // Iteration over currX
  } // Iteration over currY

  // Re-establish 1-step parent invariance for all parent IDs
  //
  // NOTE: this works since all parent IDs have smaller values than the IDs
  //       of their children, so incrementally updating the first occurrence of
  //       2-step-invariant parent from small to large IDs will re-establish
  //       1-step parent invariance for the image IDs
  for (parentID = 0; parentID < parentIDs.size(); parentID++) {
    parentIDs[parentID] = parentIDs[parentIDs[parentID]];
  }

#ifdef CONDENSE_DISJOINT_SET_LABELS
  // Count non-empty sets
  std::vector<unsigned int> setCounts(parentIDs.size(), 0);
  for (currY = 0; currY < imageIDs.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < imageIDs.cols; currX++) {
      setCounts[parentIDs[imageIDRow[currX]]] += 1;
    }
  }

  // Re-label parent IDs to be contiguous
  unsigned int freeNewID = 0;
  std::vector<unsigned int> newParentIDs;
  std::vector<unsigned char> newSetLabels;
  for (unsigned int i = 0; i < setCounts.size(); i++) {
    if (setCounts[i] > 0) {
      parentIDs[i] = freeNewID;
      newParentIDs.push_back(freeNewID);
      newSetLabels.push_back(setLabels[i]);
      freeNewID += 1;
    }
  }
#endif

#ifdef ZERO_STEP_PARENT_INVARIANCE
  // Re-establish 0-step parent invariance for all image IDs
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      imageIDRow[currX] = parentIDs[imageIDRow[currX]];
    }
  }
#endif

#ifdef CONDENSE_DISJOINT_SET_LABELS
  // Swap parent IDs and set labels
  parentIDs.swap(newParentIDs);
  setLabels.swap(newSetLabels);
#endif
};


void BaseCV::condenseDisjointSetLabels(
    cv::Mat& imageIDs,
    std::vector<unsigned int>& parentIDs,
    std::vector<unsigned char>& setLabels,
    unsigned int nStepParentInvariance) {
  int currY, currX;
  unsigned int* imageIDRow;
  std::vector<unsigned int> setCounts(parentIDs.size(), 0);
  std::vector<unsigned int> newParentIDs(parentIDs.size(), 0);
  std::vector<unsigned char> newSetLabels;

  // Count sizes of non-empty sets, and re-establish 0-step invariance
  unsigned int currParentID, iStep;
  for (currY = 0; currY < imageIDs.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < imageIDs.cols; currX++) {
      currParentID = imageIDRow[currX];
      for (iStep = 1; iStep <= nStepParentInvariance; iStep++) currParentID = parentIDs[currParentID];
      setCounts[currParentID] += 1;
    }
  }

  // Re-label parent IDs to be contiguous
  unsigned int freeNewID = 0;
  for (unsigned int i = 0; i < setCounts.size(); i++) {
    if (setCounts[i] > 0) {
      newParentIDs[i] = freeNewID;
      newSetLabels.push_back(setLabels[i]);
      freeNewID += 1;
    }
  }

  // Re-label image IDs to their new parent IDs (and thus enforce 0-step parent invariance)
  for (currY = 0; currY < imageIDs.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < imageIDs.cols; currX++) {
      imageIDRow[currX] = newParentIDs[imageIDRow[currX]];
    }
  }

  // Update parent IDs and set labels
  parentIDs.clear();
  for (unsigned int i = 0; i < freeNewID; i++) parentIDs.push_back(i);
  setLabels.swap(newSetLabels);
};


void BaseCV::debugDisjointSetLabels(
    const cv::Mat& imageIDs,
    const std::vector<unsigned int>& parentIDs,
    const std::vector<unsigned char>& setLabels) throw (const std::string&) {
  int currY, currX;
  const unsigned int* imageIDRow;
  std::vector<unsigned int> setCounts(parentIDs.size(), 0);
  std::vector<unsigned int> newParentIDs(parentIDs.size(), 0);
  std::vector<unsigned char> newSetLabels;

  // Determine counts of pixel IDs who satisfy 1-step parent invariance, and those that satisfy N-step parent invariance
  unsigned int currImageID, currParentID;
  unsigned int ONE_STEP_INVARIANCE_COUNT = 0;
  unsigned int N_STEP_INVARIANCE_COUNT = 0;
  for (currY = 0; currY < imageIDs.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < imageIDs.cols; currX++) {
      currImageID = imageIDRow[currX];
      currParentID = parentIDs[currImageID];
      if (currImageID != currParentID) {
        if (currParentID == parentIDs[currParentID]) {
          ONE_STEP_INVARIANCE_COUNT += 1;
        } else {
          N_STEP_INVARIANCE_COUNT += 1;
          while (currParentID != parentIDs[currParentID]) currParentID = parentIDs[currParentID];
        }
      }
    }
  }

  if (N_STEP_INVARIANCE_COUNT > 0) {
    ostringstream oss;
    oss << "ERROR - debugDisjointSetLabels: Found " <<
        ONE_STEP_INVARIANCE_COUNT << " 1-step parent invariances and " <<
        N_STEP_INVARIANCE_COUNT << " N-step parent invariances" << endl;
    throw oss.str();
  }

  // check whether IDs inside parentIDs == their index
  for (unsigned int i = 0; i < parentIDs.size(); i++) {
    if (parentIDs[i] != i) {
      ostringstream oss;
      oss << "ERROR - debugDisjointSetLabels: parentIDs[i] (" << parentIDs[i] <<
          ") != i (" << i << ")" << endl;
      throw oss.str();
    }
  }

  // check whether all imageIDs map to below the size of (new) parentIDs
  for (currY = 0; currY < imageIDs.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < imageIDs.cols; currX++) {
      if (imageIDRow[currX] >= parentIDs.size()) {
        ostringstream oss;
        oss << "ERROR - debugDisjointSetLabels: imageIDs[" <<
            currY << ", " << currX << "] (" << imageIDRow[currX] <<
            ") >= parentIDs.size() (" << parentIDs.size() << ")" << endl;
        throw oss.str();
      }
    }
  }
};


void BaseCV::removeIslands(
    cv::Mat& binLabelImage,
    cv::Mat& imageIDs,
    std::vector<unsigned int>& parentIDs,
    std::vector<unsigned char>& setLabels) {
  unsigned char* binLabelRow = NULL;
  unsigned int* imageIDRow = NULL;
  int currY, currX;
  unsigned int leftID, parentID, currID;

  // Assign set IDs to all pixels
  computeDisjointSets(binLabelImage, imageIDs, parentIDs, setLabels, true);

  // Identify all sets that are connected to boundary pixels
  std::vector<bool> setOnBoundary(parentIDs.size(), false);

  const unsigned int* imageIDTopRow = imageIDs.ptr<unsigned int>(0);
  const unsigned int* imageIDBottomRow = imageIDs.ptr<unsigned int>(binLabelImage.rows - 1);
  for (currX = 0; currX < binLabelImage.cols; currX++) { // Scan through top and bottom borders
    parentID = parentIDs[imageIDTopRow[currX]];
    setOnBoundary[parentID] = true;
    parentID = parentIDs[imageIDBottomRow[currX]];
    setOnBoundary[parentID] = true;
  }
  for (currY = 1; currY < binLabelImage.rows - 1; currY++) { // Scan through left and right borders
    parentID = parentIDs[imageIDs.at<unsigned int>(currY, 0)];
    setOnBoundary[parentID] = true;
    parentID = parentIDs[imageIDs.at<unsigned int>(currY, binLabelImage.cols - 1)];
    setOnBoundary[parentID] = true;
  }

#ifdef REMOVE_ISLANDS_SHOW_MERGED_DISJOINT_SETS
  imshow("Before removeIslands()", binLabelImage*255);
  cv::Mat mergedImage; // 2 clusters = 0 and 255; merged disjoint sets = 128
  binLabelImage.copyTo(mergedImage);
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    binLabelRow = mergedImage.ptr<unsigned char>(currY);
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      parentID = parentIDs[imageIDRow[currX]];
      if (setOnBoundary[parentID]) {
        binLabelRow[currX] = (setLabels[parentID] != 0) ? 255 : 0;
      } else {
        binLabelRow[currX] = 128;
      }
    }
  }
  imshow("After removeIslands()", mergedImage);
#endif

  // Merge sets that are not on the boundary to their left neighbor's sets
  // PROOF: Consider the first pixel satisfying the previous property;
  //        given that this selected pixel is the >FIRST< instance
  //        that does not belong to a boundary set, then by logic its
  //        left and top neighbors must be on the boundary set.
  //        Both of these neighbors must exist, since our iterations
  //        skip over the boundary values. After merging, apply the above
  //        algorithm iteratively for the next (a.k.a. now-first) pixel
  //        satisfying this assumption. Thus ends the inductive proof.
  //
  // NOTE: Even though we don't need to run this algorithm over the boundary
  //       pixels, we still do in order to count the size of disjoint sets
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    binLabelRow = binLabelImage.ptr<unsigned char>(currY);
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      currID = imageIDRow[currX];
      parentID = parentIDs[currID];
      if (setOnBoundary[parentID]) {
        // If parent ID is on boundary set, then re-establish 1-step parent
        // invariance, since the following else-clause breaks it
        parentIDs[currID] = parentIDs[parentID];
      } else {
        // If parent ID is not on boundary set, then merge your parent's set
        // with your left neighbor's parent's set
        leftID = parentIDs[imageIDRow[currX - 1]];
        setOnBoundary[currID] = true;
        setOnBoundary[parentID] = true;
        parentIDs[currID] = leftID;
        parentIDs[parentID] = leftID; // This breaks the 1-step parent invariance
      }
      parentID = parentIDs[currID];

      // Re-establish 0-step invariance
      // NOTE: re-establishing 0-step invariance adds ~15us (2030 -> 2045 us)
      //imageIDRow[currX] = parentID;

      // Update label to parent set's label
      binLabelRow[currX] = setLabels[parentID];
    }
  }

  return;
};


void BaseCV::removeSmallPeninsulas(
    cv::Mat& binLabelImage,
    cv::Mat& imageIDs,
    std::vector<unsigned int>& parentIDs,
    const std::vector<unsigned char>& setLabels,
    double minSetRatio) {
  const unsigned int minSetCount = minSetRatio * binLabelImage.rows * binLabelImage.cols;
  unsigned char* binLabelRow = NULL;
  unsigned int* imageIDRow = NULL;
  unsigned int* prevImageIDRow = NULL;
  int currY, currX;
  unsigned int leftID, parentID, currID;

  // Count number of entries in each set
  std::vector<unsigned int> setCounts(parentIDs.size(), 0);
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      setCounts[parentIDs[imageIDRow[currX]]] += 1;
    }
  }

  // Merge all sets whose size is below requested threshold
  //
  // NOTE: Excluding pixels belonging (directly) to the top-left set,
  //       all pixels in sets that need to be merged are merged with their
  //       left/top neighbour pixel, and all merged pixels will have
  //       0-step parent invariance. Once the entire image is scanned,
  //       and it is determined that the top-left set needs to be merged,
  //       then they are merged at that time.
  bool mergeTopLeft = false;
  unsigned int topLeftParentID = 0;
  unsigned int topLeftSetNeighborID = 0;
  bool topLeftSetNeighborIDFound = false;
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    prevImageIDRow = imageIDRow;
    binLabelRow = binLabelImage.ptr<unsigned char>(currY);
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      currID = imageIDRow[currX];
      parentID = parentIDs[parentIDs[currID]]; // 1-step parent invariance is broken by clause below
      if (setCounts[parentID] > 0 && setCounts[parentID] < minSetCount) { // Merge all sets whose size is below threshold
        // Don't merge a member of the top-left set with anything else until after scanning through entire image, to preserve consistency of disjoint set
        if (mergeTopLeft && parentID == topLeftParentID) {
          leftID = parentID;
        }
        // Merge your parent's set with your left or top neighbor's parent's set
        else if (currX == 0) {
          if (currY == 0) { // If top-left pixel is a marked set, merge with bottom or right neighbor (at the end of the scan)
            mergeTopLeft = true;
            topLeftParentID = parentID;
            leftID = parentID;
          } else { // Entry is on left column, but is not top-left pixel
            leftID = parentIDs[parentIDs[prevImageIDRow[currX]]]; // 1-step parent invariance is broken by clause below
          }
        } else { // Entry is not on left column
          leftID = parentIDs[parentIDs[imageIDRow[currX - 1]]]; // 1-step parent invariance is broken by clause below
        }
        imageIDRow[currX] = leftID;
        parentIDs[currID] = leftID; // This breaks 0-step parent invariance for other pixels whose parentIDs[currID] == parentIDs[imageIDs_ij] != leftID
        parentIDs[parentID] = leftID; // This breaks 1-step parent invariance for other pixels whose parentIDs[parentID] == parentIDs[imageIDs_ij] != leftID
        parentID = leftID;
      } else { // Sets that do not need to be merged
        // Re-establish 1-step invariance, since pixel may belong to a previously-merged set
        parentIDs[currID] = parentID;

        // Identify the dominant neighbor set of the top left set
        if (mergeTopLeft && !topLeftSetNeighborIDFound) {
          if (currX != 0) {
            if (parentIDs[parentIDs[imageIDRow[currX-1]]] == topLeftParentID) {
              topLeftSetNeighborID = parentID;
              topLeftSetNeighborIDFound = true;
            }
          }
          if (currY != 0) {
            if (parentIDs[parentIDs[prevImageIDRow[currX]]] == topLeftParentID) {
              topLeftSetNeighborID = parentID;
              topLeftSetNeighborIDFound = true;
            }
          }
        }
      }

      // Update label to parent set's label
      binLabelRow[currX] = setLabels[parentID];

      // Re-establish 0-step invariance
      // NOTE: re-establishing 0-step invariance adds ~45us (2675 -> 2720 us)
      //imageIDRow[currX] = parentID;
    } // for each column
  } // for each row

  if (mergeTopLeft) {
    if (topLeftSetNeighborIDFound) {
      // Update all entries belonging to the top-left set
      const unsigned char topLeftSetNeighborLabel = setLabels[topLeftSetNeighborID];
      bool foundInRow = false;
      for (currY = 0; currY < binLabelImage.rows; currY++) {
        binLabelRow = binLabelImage.ptr<unsigned char>(currY);
        imageIDRow = imageIDs.ptr<unsigned int>(currY);
        foundInRow = false;
        for (currX = 0; currX < binLabelImage.cols; currX++) {
          if (parentIDs[imageIDRow[currX]] == topLeftParentID) {
            binLabelRow[currX] = topLeftSetNeighborLabel;
            foundInRow = true;

            // NOTE: these image IDs need to be re-established to 0-step
            //       invariance, since their old (top-left) parent ID is
            //       guaranteed to be smaller than their new parent ID, so
            //       cannot just map parentIDs[0] == newParentID
            imageIDRow[currX] = topLeftSetNeighborID;
          }
        }
        if (!foundInRow) { break; }
      }

      // Check how many non-zero-sized parent IDs have links to the current top-left parent ID (which implies non-1-step invariance)
      //parentIDs[topLeftParentID] = topLeftSetNeighborID; // WARNING: this violates invariance that parent of a given ID is smaller than it
    } // Else this signifies that all sets have merged to top-left set, so no need to merge top-left set again
    // NOTE: technically we should flip all the labels once, but we assume that for a single-labelled image
    //       it does not matter what label it is
  } // if mergeTopLeft

  return;
};


void BaseCV::removePeninsulas(
    cv::Mat& binLabelImage,
    cv::Mat& imageIDs,
    std::vector<unsigned int>& parentIDs,
    const std::vector<unsigned char>& setLabels,
    int boundaryType) {
  unsigned char* binLabelRow = NULL;
  unsigned int* imageIDRow = NULL;
  unsigned int* prevImageIDRow = NULL;
  int currY, currX;
  unsigned int leftID, parentID, currID;

  // Count number of entries in each set
  std::vector<unsigned int> setCounts(parentIDs.size(), 0);
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      setCounts[parentIDs[imageIDRow[currX]]] += 1;
    }
  }

  // Determine the number of sets, and also the largest set with label = 0
  // and one with label = 1
  unsigned int numSets = 0;
  unsigned int label0MaxID = -1, label1MaxID = -1; // setting default to -1 to force no-match
  unsigned int label0SecMaxID = -1, label1SecMaxID = -1; // (unless there are a total of 4294967295 IDs, i.e. 65k x 65k image)
  unsigned int label0MaxSize = 0, label1MaxSize = 0;
  unsigned int label0SecMaxSize = 0, label1SecMaxSize = 0;
  bool label0MaxFound = false, label1MaxFound = false;
  //bool label0SecMaxFound = false, label1SecMaxFound = false; // UNUSED
  for (unsigned int i = 0; i < setCounts.size(); i++) {
    if (setCounts[i] > 0) {
      numSets += 1;
      if (setLabels[i] != 0) { // 1-label
        if (label1MaxFound) {
          //label1SecMaxFound = true; // UNUSED
          if (setCounts[i] > label1MaxSize) {
            label1SecMaxID = label1MaxID;
            label1SecMaxSize = label1MaxSize;
            label1MaxID = i;
            label1MaxSize = setCounts[i];
          } else if (setCounts[i] > label1SecMaxSize) {
            label1SecMaxID = i;
            label1SecMaxSize = setCounts[i];
          }
        } else {
          label1MaxFound = true;
          if (setCounts[i] > label1MaxSize) {
            label1MaxID = i;
            label1MaxSize = setCounts[i];
          }
        }
      } else { // 0-label
        if (label0MaxFound) {
          //label0SecMaxFound = true; // UNUSED
          label0MaxFound = true;
          if (setCounts[i] > label0MaxSize) {
            label0SecMaxID = label0MaxID;
            label0SecMaxSize = label0MaxSize;
            label0MaxID = i;
            label0MaxSize = setCounts[i];
          } else if (setCounts[i] > label0SecMaxSize) {
            label0SecMaxID = i;
            label0SecMaxSize = setCounts[i];
          }
        } else {
          label0MaxFound = true;
          if (setCounts[i] > label0MaxSize) {
            label0MaxID = i;
            label0MaxSize = setCounts[i];
          }
        }
      }
    }
  }

  // Determine which labels to NOT be merged, depending on boundary type
  if (boundaryType == EDGE_BOUNDARY) {
    // Keep only the single largest set per label
    label0SecMaxID = -1;
    label1SecMaxID = -1;
  } else if (boundaryType == STRIP_BOUNDARY) {
    // Keep both pairs of sets for STRIP_BOUNDARY
    /*
    // Keep both largest sets per label, and also keep the second largest set
    // for the label whose first + second largest set count is largest
    if (label0SecMaxFound) {
      if (label1SecMaxFound) { // Both second largest sets found
        if (label0MaxSize + label0SecMaxSize > label1MaxSize + label1SecMaxSize) {
          label1SecMaxID = -1;
        } else {
          label0SecMaxID = -1;
        }
      } else { // Only the second largest set for 0-label is found, so keep by default
        label1SecMaxID = -1; // for redundancy
      }
    } else if (label1SecMaxFound) { // Only the second largest set for 1-label is found, so keep by default
        label0SecMaxID = -1; // for redundancy
    } // else neither second largest sets found, which means that only 2 sets exist, so stop (already caught above)
    */
  } // For any unrecognized boundary types, by default keep all 4 largest sets

  // Merge all sets whose IDs are not equal to any of the marked 4 IDs
  //
  // NOTE: Excluding pixels belonging (directly) to the top-left set,
  //       all pixels in sets that need to be merged are merged with their
  //       left/top neighbour pixel, and all merged pixels will have
  //       0-step parent invariance. Once the entire image is scanned,
  //       and it is determined that the top-left set needs to be merged,
  //       then they are merged at that time.
  bool mergeTopLeft = false;
  unsigned int topLeftParentID = 0;
  unsigned int topLeftSetNeighborID = 0;
  bool topLeftSetNeighborIDFound = false;
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    prevImageIDRow = imageIDRow;
    binLabelRow = binLabelImage.ptr<unsigned char>(currY);
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      currID = imageIDRow[currX];
      parentID = parentIDs[parentIDs[currID]]; // 1-step parent invariance is broken by clause below


      if (!((parentID == label0MaxID) || (parentID == label1MaxID) ||
          (parentID == label0SecMaxID) || (parentID == label1SecMaxID))) { // Merge all non-marked sets
        // Don't merge a member of the top-left set with anything else until after scanning through entire image, to preserve consistency of disjoint set
        if (mergeTopLeft && parentID == topLeftParentID) {
          leftID = parentID;
        }
        // Merge your parent's set with your left or top neighbor's parent's set
        else if (currX == 0) {
          if (currY == 0) { // If top-left pixel is a marked set, merge with bottom or right neighbor (at the end of the scan)
            mergeTopLeft = true;
            topLeftParentID = parentID;
            leftID = parentID;
          } else { // Entry is on left column, but is not top-left pixel
            leftID = parentIDs[parentIDs[prevImageIDRow[currX]]]; // 1-step parent invariance is broken by clause below
          }
        } else { // Entry is not on left column
          leftID = parentIDs[parentIDs[imageIDRow[currX - 1]]]; // 1-step parent invariance is broken by clause below
        }
        imageIDRow[currX] = leftID;
        parentIDs[currID] = leftID; // This breaks 0-step parent invariance for other pixels whose parentIDs[currID] == parentIDs[imageIDs_ij] != leftID
        parentIDs[parentID] = leftID; // This breaks 1-step parent invariance for other pixels whose parentIDs[parentID] == parentIDs[imageIDs_ij] != leftID
        parentID = leftID;
      } else { // Sets that do not need to be merged
        // Re-establish 1-step invariance, since pixel may belong to a previously-merged set
        parentIDs[currID] = parentID;

        // Identify the dominant neighbor set of the top left set
        if (mergeTopLeft && !topLeftSetNeighborIDFound) {
          if (currX != 0) {
            if (parentIDs[parentIDs[imageIDRow[currX-1]]] == topLeftParentID) {
              topLeftSetNeighborID = parentID;
              topLeftSetNeighborIDFound = true;
            }
          }
          if (currY != 0) {
            if (parentIDs[parentIDs[prevImageIDRow[currX]]] == topLeftParentID) {
              topLeftSetNeighborID = parentID;
              topLeftSetNeighborIDFound = true;
            }
          }
        }
      }

      // Update label to parent set's label
      binLabelRow[currX] = setLabels[parentID];

      // Re-establish 0-step invariance
      // NOTE: re-establishing 0-step invariance adds ~45us (2675 -> 2720 us)
      //imageIDRow[currX] = parentID;
    } // for each column
  } // for each row

  if (mergeTopLeft) {
    if (topLeftSetNeighborIDFound) {
      // Update all entries belonging to the top-left set
      const unsigned char topLeftSetNeighborLabel = setLabels[topLeftSetNeighborID];
      bool foundInRow = false;
      for (currY = 0; currY < binLabelImage.rows; currY++) {
        binLabelRow = binLabelImage.ptr<unsigned char>(currY);
        imageIDRow = imageIDs.ptr<unsigned int>(currY);
        foundInRow = false;
        for (currX = 0; currX < binLabelImage.cols; currX++) {
          if (parentIDs[imageIDRow[currX]] == topLeftParentID) {
            binLabelRow[currX] = topLeftSetNeighborLabel;
            foundInRow = true;

            // NOTE: these image IDs need to be re-established to 0-step
            //       invariance, since their old (top-left) parent ID is
            //       guaranteed to be smaller than their new parent ID, so
            //       cannot just map parentIDs[0] == newParentID
            imageIDRow[currX] = topLeftSetNeighborID;
          }
        }
        if (!foundInRow) { break; }
      }

      // Check how many non-zero-sized parent IDs have links to the current top-left parent ID (which implies non-1-step invariance)
      //parentIDs[topLeftParentID] = topLeftSetNeighborID; // WARNING: this violates invariance that parent of a given ID is smaller than it
    } // Else this signifies that all sets have merged to top-left set, so no need to merge top-left set again
    // NOTE: technically we should flip all the labels once, but we assume that for a single-labelled image
    //       it does not matter what label it is
  } // if mergeTopLeft

  return;
};


void BaseCV::findConnectedPixels(
    const cv::Mat& binLabelImage,
    const cv::Point targetPixel,
    std::vector<cv::Point>& connectedSet,
    bool eightConnected) {
  cv::Mat imageIDs(binLabelImage.size(), CV_32SC1);
  // NOTE: even though we are specifying type == signed int, we can still be
  //       self-consistent and use it as unsigned int
  std::vector<unsigned int> parentIDs;
  std::vector<unsigned char> setLabels;
  int currY, currX;
  unsigned int currID;

  // Assign set IDs to all pixels
  computeDisjointSets(binLabelImage, imageIDs, parentIDs, setLabels, eightConnected);

  // Since computeDisjointSets enforces 1-step parent invariance, we can just
  // select all pixels whose parentIDs[imageID] is equal to the parentIDs[target's]
  CV_Assert(targetPixel.x >= 0 && targetPixel.x < binLabelImage.cols &&
      targetPixel.y >= 0 && targetPixel.y < binLabelImage.rows);
  unsigned int targetID = parentIDs[imageIDs.at<unsigned int>(targetPixel.y, targetPixel.x)];
  connectedSet.clear();
  for (currY = 0; currY < binLabelImage.rows; currY++) {
    const unsigned int* imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < binLabelImage.cols; currX++) {
      currID = parentIDs[imageIDRow[currX]];
      if (currID == targetID) {
        connectedSet.push_back(cv::Point(currX, currY));
      }
    }
  }
};


void BaseCV::applySobel(
    const cv::Mat& binImage,
    const cv::Mat& mask,
    std::vector<cv::Point>& edgePoints,
    cv::Mat& edgelBuffer,
    bool fillEdgelBuffer) {
  cv::Mat sobelX, sobelY;
  cv::Sobel(binImage, sobelX, CV_16S, 1, 0, 3);
  cv::Sobel(binImage, sobelY, CV_16S, 0, 1, 3);
  cv::multiply(sobelX, sobelX, sobelX);
  cv::multiply(sobelY, sobelY, sobelY);
  sobelX = sobelX + sobelY; // = squared Sobel magnitude maximum 3x3 Sobel response value = 20

#ifdef APPLY_SOBEL_VIEW_MAG
  cv::Mat sobelMag;
  sobelX.convertTo(sobelMag, CV_8U, 255.0/20, 0);
  imshow("Sobel Edge Magnitude", sobelMag);
#endif

#ifdef APPLY_SOBEL_VIEW_OUTPUT
  imshow("Thresholded Sobel Edge Response", (sobelX >= SOBEL_MAG_SQRD_THRESHOLD));
#endif

  edgePoints.clear();
  int currY, currX;
  if (mask.empty()) {
    for (currY = 0; currY < binImage.rows; currY++) {
      const short* currRow = sobelX.ptr<short>(currY);
      for (currX = 0; currX < binImage.cols; currX++) {
        if (currRow[currX] >= SOBEL_MAG_SQRD_THRESHOLD) {
          edgePoints.push_back(cv::Point(currX, currY));
        }
      }
    }

    if (fillEdgelBuffer) {
      edgelBuffer = (sobelX >= SOBEL_MAG_SQRD_THRESHOLD);
    }
  } else {
    for (currY = 0; currY < binImage.rows; currY++) {
      const short* currRow = sobelX.ptr<short>(currY);
      for (currX = 0; currX < binImage.cols; currX++) {
        if (currRow[currX] >= SOBEL_MAG_SQRD_THRESHOLD &&
            mask.at<unsigned char>(currY, currX) != 0) {
          edgePoints.push_back(cv::Point(currX, currY));
        }
      }
    }

    if (fillEdgelBuffer) {
      edgelBuffer = (sobelX >= SOBEL_MAG_SQRD_THRESHOLD) & mask;
    }
  }
};


double BaseCV::solveLineRANSAC(
    const std::vector<cv::Point>& points,
    cv::Vec4f& line,
    double goodFitDist,
    unsigned int baseLineFitCount,
    unsigned int maxNumIters,
    double candidateFitRatio,
    double termFitRatio) {
  unsigned int currIter = 0, pointI = 0, updatedConsensusCount;
  double currSizeRatio, bestSizeRatio;
  double bestError = std::numeric_limits<double>::infinity();
  double currError = bestError, currDist = 0;
  vector<unsigned int> IDs;
  vector<cv::Point> initPoints;
  vector<cv::Point> consensusPoints;
  cv::Vec4f currFit, bestFit;
  cv::Point pointA, pointB;
  bool hasBest = false;
#ifdef SHOW_RANSAC_FINAL
  vector<cv::Point> bestConsensus;
#endif

  // Validate input parameters
  CV_Assert(candidateFitRatio >= 0 && candidateFitRatio <= 1);
  CV_Assert(termFitRatio >= 0 && termFitRatio <= 1);
  if (points.size() <= 1) { // Not enough points to draw line
    line = cv::Vec4f(0, 0, 0, 0);
    return std::numeric_limits<double>::infinity();
  } else if (points.size() < baseLineFitCount) {
    cv::fitLine(cv::Mat(points), line, CV_DIST_L2, 0, \
        RANSAC_LINE_FIT_RADIUS_EPS, RANSAC_LINE_FIT_ANGLE_EPS);
    std::vector<cv::Point>::const_iterator itPoints = points.begin();
    std::vector<cv::Point>::const_iterator itPointsEnd = points.end();
    currError = 0;
    for (; itPoints != itPointsEnd; itPoints++) {
      currError += distPointLine(*itPoints, line);
    }
    currError = currError / points.size() / points.size();
    return currError;
  }

  for (pointI = 0; pointI < points.size(); pointI++) {
    IDs.push_back(pointI);
  }

  while (currIter < maxNumIters) {
    // Select an initial random set of points and fit line to it
    random_shuffle(IDs.begin(), IDs.end());
    if (baseLineFitCount == 2) {
      pointA = points[IDs[0]], pointB = points[IDs[1]];
      currFit[2] = pointA.x; // Set x0
      currFit[3] = pointA.y; // Set y0
      if (pointA.x == pointB.x && pointA.y == pointB.y) {
        currFit[0] = 1; // Set vx to some default value
        currFit[1] = 0; // Set vy to some default value
      } else {
        currFit[0] = ((float) pointB.x) - pointA.x;
        currFit[1] = ((float) pointB.y) - pointA.y;
      }
    } else {
      initPoints.clear();
      for (pointI = 0; pointI < baseLineFitCount; pointI++) {
        initPoints.push_back(points[IDs[pointI]]);
      }
      cv::fitLine(cv::Mat(initPoints), currFit, CV_DIST_L2, 0, \
          RANSAC_LINE_FIT_RADIUS_EPS, RANSAC_LINE_FIT_ANGLE_EPS);
    }

    consensusPoints.clear();
    currError = 0;
    for (pointI = 0; pointI < points.size(); pointI++) {
      currDist = distPointLine(points[pointI], currFit);
      if (currDist <= goodFitDist) {
        consensusPoints.push_back(points[pointI]);
        currError += currDist;
      }
    }
    if (consensusPoints.size() <= 0) { continue; }

#ifdef SOLVE_LINE_RANSAC_PROFILE
    ransac_first_fit_avg_dist.push_back(currError / consensusPoints.size());
#endif

    currError = currError / consensusPoints.size() / consensusPoints.size(); // Divide twice to penalize smaller consensus sets
    currSizeRatio = ((double) consensusPoints.size())/points.size();

#ifdef SOLVE_LINE_RANSAC_PROFILE
    ransac_first_fit_ratio.push_back(currSizeRatio);
#endif

#ifdef SHOW_RANSAC_ITER
    cv::Mat foo = cv::Mat::zeros(480, 640, CV_8U);
    for (unsigned int ii = 0; ii < consensusPoints.size(); ii++) {
      foo.at<unsigned char>(consensusPoints[ii].y, consensusPoints[ii].x) = 0x55;
    }
    foo.at<unsigned char>(pointA.y, pointA.x) = 0xFF;
    foo.at<unsigned char>(pointB.y, pointB.x) = 0xFF;
    cv::imshow("TEMP", foo);
    cv::waitKey(0);
#endif

    // First time
    if (!hasBest) {
      hasBest = true;
      bestFit = currFit;
      bestError = currError;
      bestSizeRatio = currSizeRatio;
#ifdef SHOW_RANSAC_FINAL
      bestConsensus = consensusPoints;
#endif
    } else if (currSizeRatio >= candidateFitRatio) {
      // Fit line to consensus set
      cv::fitLine(cv::Mat(consensusPoints), currFit, CV_DIST_L2, 0, \
          RANSAC_LINE_FIT_RADIUS_EPS, RANSAC_LINE_FIT_ANGLE_EPS);

      // Compute error and update best model if necessary
      currError = 0;
      updatedConsensusCount = 0;
      for (pointI = 0; pointI < points.size(); pointI++) {
        currDist = distPointLine(points[pointI], currFit);
        if (currDist <= goodFitDist) {
          currError += currDist;
          updatedConsensusCount++;
        }
      }
      if (updatedConsensusCount <= 0) { continue; }

#ifdef SOLVE_LINE_RANSAC_PROFILE
      ransac_second_fit_avg_dist.push_back(currError / consensusPoints.size());
#endif

      currError = currError / updatedConsensusCount / updatedConsensusCount; // Divide twice to penalize smaller consensus sets
      currSizeRatio = ((double) updatedConsensusCount) / points.size();

#ifdef SOLVE_LINE_RANSAC_PROFILE
      ransac_second_fit_ratio.push_back(currSizeRatio);
#endif

      if (currError <= bestError) {
        bestFit = currFit;
        bestError = currError;
        bestSizeRatio = currSizeRatio;
#ifdef SHOW_RANSAC_FINAL
        bestConsensus = consensusPoints;
#endif

        // Terminate prematurely if accepted point ratio exceeds candidateFitRatio
        if (bestSizeRatio >= candidateFitRatio) {
#ifdef SOLVE_LINE_RANSAC_PROFILE
          ransac_term_via_fit++;
#endif
          break;
        }
      }
    }
    currIter += 1;
  }
  // cout << "-> RANSAC TERMINATED W/ " << bestSizeRatio*100 << "% & Error=" << bestError << endl << flush;
  // cout << " VX=" << bestFit[0] << " VY=" << bestFit[1] << " X0=" << bestFit[2] << " Y0=" << bestFit[3] << endl << flush;

#ifdef SOLVE_LINE_RANSAC_PROFILE
  if (currIter >= maxNumIters) {
    ransac_term_via_iters++;
  }
#endif

#ifdef SHOW_RANSAC_FINAL
  cv::Mat showMe = cv::Mat::zeros(480, 640, CV_8U);
  for (unsigned int iii = 0; iii < bestConsensus.size(); iii++) {
    showMe.at<unsigned char>(bestConsensus[iii].y, bestConsensus[iii].x) = 0xFF;
  }
  imshow("SHOW ME", showMe);
#endif

#ifdef SOLVE_LINE_RANSAC_PROFILE
  ransac_iters.push_back(currIter);
#endif

#ifdef SOLVE_LINE_RANSAC_PROFILE
  std::cout << "====================" << std::endl;
  if (ransac_iters.size() > 1) {
    std::vector<unsigned int>::iterator itRI = ransac_iters.begin();
    std::vector<unsigned int>::iterator itRIEnd = ransac_iters.end();
    double ransac_sum = 0;
    unsigned int ransac_max = numeric_limits<unsigned int>::min();
    unsigned int ransac_min = numeric_limits<unsigned int>::max();
    for (; itRI != itRIEnd; itRI++) {
      ransac_sum += *itRI;
      if (*itRI > ransac_max) { ransac_max = *itRI; }
      if (*itRI < ransac_min) { ransac_min = *itRI; }
    }
    std::cout << "RANSAC_ITERS MIN-MEAN-MAX: " << ransac_min << ", " << \
        ransac_sum/ransac_iters.size() << ", " << ransac_max << std::endl;
  }
  if (ransac_first_fit_ratio.size() > 1) {
    std::vector<double>::iterator itRFFR = ransac_first_fit_ratio.begin();
    std::vector<double>::iterator itRFFREnd = ransac_first_fit_ratio.end();
    double ransac_sum = 0;
    double ransac_max = numeric_limits<double>::min();
    double ransac_min = numeric_limits<double>::max();
    for (; itRFFR != itRFFREnd; itRFFR++) {
      ransac_sum += *itRFFR;
      if (*itRFFR > ransac_max) { ransac_max = *itRFFR; }
      if (*itRFFR < ransac_min) { ransac_min = *itRFFR; }
    }
    std::cout << "RANSAC_FIRST_FIT_RATIO MIN-MEAN-MAX: " << ransac_min << ", " << \
        ransac_sum/ransac_first_fit_ratio.size() << ", " << ransac_max << std::endl;
  }
  if (ransac_second_fit_ratio.size() > 1) {
    std::vector<double>::iterator itRSFR = ransac_second_fit_ratio.begin();
    std::vector<double>::iterator itRSFREnd = ransac_second_fit_ratio.end();
    double ransac_sum = 0;
    double ransac_max = numeric_limits<double>::min();
    double ransac_min = numeric_limits<double>::max();
    for (; itRSFR != itRSFREnd; itRSFR++) {
      ransac_sum += *itRSFR;
      if (*itRSFR > ransac_max) { ransac_max = *itRSFR; }
      if (*itRSFR < ransac_min) { ransac_min = *itRSFR; }
    }
    std::cout << "RANSAC_SECOND_FIT_RATIO MIN-MEAN-MAX: " << ransac_min << ", " << \
        ransac_sum/ransac_second_fit_ratio.size() << ", " << ransac_max << std::endl;
  }
  if (ransac_first_fit_avg_dist.size() > 1) {
    std::vector<double>::iterator itRFFAD = ransac_first_fit_avg_dist.begin();
    std::vector<double>::iterator itRFFADEnd = ransac_first_fit_avg_dist.end();
    double ransac_sum = 0;
    double ransac_max = numeric_limits<double>::min();
    double ransac_min = numeric_limits<double>::max();
    for (; itRFFAD != itRFFADEnd; itRFFAD++) {
      ransac_sum += *itRFFAD;
      if (*itRFFAD > ransac_max) { ransac_max = *itRFFAD; }
      if (*itRFFAD < ransac_min) { ransac_min = *itRFFAD; }
    }
    std::cout << "RANSAC_FIRST_FIT_AVG_DIST MIN-MEAN-MAX: " << ransac_min << ", " << \
        ransac_sum/ransac_first_fit_avg_dist.size() << ", " << ransac_max << std::endl;
  }
  if (ransac_second_fit_avg_dist.size() > 1) {
    std::vector<double>::iterator itRSFAD = ransac_second_fit_avg_dist.begin();
    std::vector<double>::iterator itRSFADEnd = ransac_second_fit_avg_dist.end();
    double ransac_sum = 0;
    double ransac_max = numeric_limits<double>::min();
    double ransac_min = numeric_limits<double>::max();
    for (; itRSFAD != itRSFADEnd; itRSFAD++) {
      ransac_sum += *itRSFAD;
      if (*itRSFAD > ransac_max) { ransac_max = *itRSFAD; }
      if (*itRSFAD < ransac_min) { ransac_min = *itRSFAD; }
    }
    std::cout << "RANSAC_SECOND_FIT_AVG_DIST MIN-MEAN-MAX: " << ransac_min << ", " << \
        ransac_sum/ransac_second_fit_avg_dist.size() << ", " << ransac_max << std::endl;
  }
  if (ransac_term_via_iters + ransac_term_via_fit > 0) {
    std::cout << "RANSAC NUM TERMS: " << \
        ransac_term_via_iters + ransac_term_via_fit << ", VIA ITERS: " << \
        ransac_term_via_iters*100.0/(ransac_term_via_iters+ransac_term_via_fit) << \
        "%, VIA GOOD FIT: " << \
        ransac_term_via_fit*100.0/(ransac_term_via_iters+ransac_term_via_fit) << \
        "%" << std::endl;
  }
  std::cout << "--------------------" << std::endl << std::flush;
#endif

  line = bestFit;
  return bestError;
};


Contour BaseCV::findMLELargestBlobContour(Mat& src) {
  ContourVector contours;
  Contour bestContour;
  Point2f blobCenter;
  double largestBlobArea = -1;
  double currBlobArea;

  findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  for (ContourVector::iterator it = contours.begin(); it != contours.end(); it++) {
    currBlobArea = contourArea(*it);
    if (currBlobArea > largestBlobArea) {
      largestBlobArea = currBlobArea;
      bestContour = *it;
    }
  }
  return bestContour;
};


Contour BaseCV::findMLECircularBlobContour(Mat& src) {
  ContourVector contours;
  Contour bestContour;
  Point2f currCircleCenter;
  double bestBlobRank = -1;
  double currBlobRank;
  double currHullArea;
  double currEnclosingCircleArea;
  float currCircleRadius;
  std::vector<cv::Point> currConvexHull;

  findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  for (ContourVector::iterator it = contours.begin(); it != contours.end(); it++) {
    minEnclosingCircle(*it, currCircleCenter, currCircleRadius);
    convexHull(*it, currConvexHull);
    currHullArea = contourArea(currConvexHull);
    currEnclosingCircleArea = M_PI*currCircleRadius*currCircleRadius;
    currBlobRank = pow(currHullArea / currEnclosingCircleArea, 10)*currCircleRadius;
    if (currBlobRank > bestBlobRank) {
      bestBlobRank = currBlobRank;
      bestContour = *it;
    }
  }
  return bestContour;
};


Point2f BaseCV::findMLECircularBlobCenter(Mat& src, blob::Contour& bestContourBuffer) {
  ContourVector contours;
  Point2f bestCircleCenter(-1, -1);
  Point2f currCircleCenter;
  double bestBlobRank = -1;
  double currBlobRank;
  double currHullArea;
  double currEnclosingCircleArea;
  float currCircleRadius;
  std::vector<cv::Point> currConvexHull;

  findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  ContourVector::iterator bestContourIt;
  for (ContourVector::iterator it = contours.begin(); it != contours.end(); it++) {
    minEnclosingCircle(*it, currCircleCenter, currCircleRadius);
    convexHull(*it, currConvexHull);
    currHullArea = contourArea(currConvexHull);
    currEnclosingCircleArea = M_PI*currCircleRadius*currCircleRadius;
    currBlobRank = pow(currHullArea / currEnclosingCircleArea, 10)*currCircleRadius;
    if (currBlobRank > bestBlobRank) {
      bestBlobRank = currBlobRank;
      bestCircleCenter = currCircleCenter;
      bestContourIt = it;
    }
  }
  bestContourBuffer = *bestContourIt;

  return bestCircleCenter;
};


void BaseCV::applyGrayscaleThresholdClassifier(const Mat& src, Mat& dst, char thresh) {
  if (src.channels() != 1) {
    cv::Mat grayscaleSrc;
    cvtColor(src, grayscaleSrc, CV_RGB2GRAY);
    threshold(grayscaleSrc, dst, (double) thresh, 255, THRESH_BINARY);
  } else {
    threshold(src, dst, (double) thresh, 255, THRESH_BINARY);
  }
};


void BaseCV::applyHueRangeClassifier(const Mat& src, Mat& dst,
    int minHueDeg, int maxHueDeg, double medianBlurWidthRatio) {
  Mat hsv, hue;
  cvtColor(src, hsv, CV_BGR2HSV);
  hue.create(hsv.size(), hsv.depth());
  int ch[] = {0, 0};
  mixChannels(&hsv, 1, &hue, 1, ch, 1);

  dst.create(hue.size(), CV_8U);
  minHueDeg = wrapAngle(minHueDeg)/2.0;
  maxHueDeg = wrapAngle(maxHueDeg)/2.0;
  if (minHueDeg <= maxHueDeg) {
    inRange(hue, Scalar(minHueDeg), Scalar(maxHueDeg), dst);
  } else {
    Mat tmp(hue.size(), CV_8U);
    inRange(hue, Scalar(0), Scalar(maxHueDeg), tmp);
    inRange(hue, Scalar(minHueDeg), Scalar(180), dst);
    dst.mul(tmp);
  }
  try {
    medianBlurWidthRatio = min(1.0, max(0.0, medianBlurWidthRatio));
    medianBlur(dst, dst, (int) floor(max(2.0, min(dst.rows, dst.cols)*medianBlurWidthRatio)/2)*2 + 1);
  } catch (Exception& err) {
    throw string(err.what());
  }
};


Point2f BaseCV::findBoundingCircleCenter(const Contour& ctr) {
  Point2f center;
  float radius;
  minEnclosingCircle(ctr, center, radius);
  return center;
};


void BaseCV::drawArrow(cv::Mat& bgrBuffer, double headingDeg,
    cv::Scalar edgeColor, cv::Scalar fillColor, double alphaRatio) {
  if (headingDeg == vc_math::INVALID_ANGLE) return;

  double arrowLineLength = std::min(bgrBuffer.cols, bgrBuffer.rows)*0.4;
  double arrowEarLength = std::max(arrowLineLength*0.22, 10.0);
  unsigned int arrowLineWidth = std::max(int(arrowLineLength/8.0), 6);
  cv::Point midPoint = cv::Point(bgrBuffer.cols/2, bgrBuffer.rows/2);
  cv::Point tipPoint = midPoint;
  tipPoint.x += sin(headingDeg*degree)*arrowLineLength;
  tipPoint.y -= cos(headingDeg*degree)*arrowLineLength;
  cv::Point leftEarPoint = tipPoint;
  headingDeg += 180 - 40;
  leftEarPoint.x += sin(headingDeg*degree)*arrowEarLength;
  leftEarPoint.y -= cos(headingDeg*degree)*arrowEarLength;
  cv::Point rightEarPoint = tipPoint;
  headingDeg += 80;
  rightEarPoint.x += sin(headingDeg*degree)*arrowEarLength;
  rightEarPoint.y -= cos(headingDeg*degree)*arrowEarLength;

  if (alphaRatio >= 1.0 || alphaRatio <= 0.0) {
    arrowLineWidth *= 1.5;
    line(bgrBuffer, midPoint, tipPoint, edgeColor, arrowLineWidth, 8, 0);
    line(bgrBuffer, tipPoint, leftEarPoint, edgeColor, arrowLineWidth, 8, 0);
    line(bgrBuffer, tipPoint, rightEarPoint, edgeColor, arrowLineWidth, 8, 0);
    arrowLineWidth /= 2;
    line(bgrBuffer, midPoint, tipPoint, fillColor, arrowLineWidth, 8, 0);
    line(bgrBuffer, tipPoint, leftEarPoint, fillColor, arrowLineWidth, 8, 0);
    line(bgrBuffer, tipPoint, rightEarPoint, fillColor, arrowLineWidth, 8, 0);
  } else {
    cv::Mat canvas;
    bgrBuffer.copyTo(canvas);

    arrowLineWidth *= 1.5;
    line(canvas, midPoint, tipPoint, edgeColor, arrowLineWidth, 8, 0);
    line(canvas, tipPoint, leftEarPoint, edgeColor, arrowLineWidth, 8, 0);
    line(canvas, tipPoint, rightEarPoint, edgeColor, arrowLineWidth, 8, 0);
    arrowLineWidth /= 2;
    line(canvas, midPoint, tipPoint, fillColor, arrowLineWidth, 8, 0);
    line(canvas, tipPoint, leftEarPoint, fillColor, arrowLineWidth, 8, 0);
    line(canvas, tipPoint, rightEarPoint, fillColor, arrowLineWidth, 8, 0);

    addWeighted(canvas, alphaRatio, bgrBuffer, (1.0 - alphaRatio), 0, bgrBuffer);
  }
};
