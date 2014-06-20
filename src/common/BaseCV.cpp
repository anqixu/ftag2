#include "common/BaseCV.hpp"
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/math/constants/constants.hpp>


#define USE_OPTIMIZED_SIN_COS_IN_KMEANS


using namespace std;
using namespace cv;
using namespace vc_math;


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
#ifdef SOLVE_LINE_RANSAC_PROFILE
std::vector<unsigned int> BaseCV::ransac_iters;
std::vector<double> BaseCV::ransac_first_fit_ratio;
std::vector<double> BaseCV::ransac_second_fit_ratio;
std::vector<double> BaseCV::ransac_first_fit_avg_dist;
std::vector<double> BaseCV::ransac_second_fit_avg_dist;
unsigned int BaseCV::ransac_term_via_iters = 0;
unsigned int BaseCV::ransac_term_via_fit = 0;
#endif


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
      for (iStep = 1; iStep <= nStepParentInvariance; iStep++) {
        if (currParentID == parentIDs[currParentID]) break;
        currParentID = parentIDs[currParentID];
      }
      imageIDRow[currX] = currParentID;
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


void BaseCV::rotate90(const cv::Mat& src, cv::Mat& dst, int K) {
  K = K % 4;
  if (K == 0) {
    src.copyTo(dst);
  } else if (K == 1) { // Rotate 90' counter-clockwise
    dst = src.t();
    cv::flip(dst, dst, 0);
  } else if (K == 2) { // Rotate 180'
    cv::flip(src, dst, -1);
  } else if (K == 3) { // Rotate 90' clockwise
    cv::flip(src, dst, 0);
    dst = dst.t();
  }
};
