#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{   

    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
 
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {   
        if (descSource.type() != CV_32F || descRef.type() != CV_32F)
        { 
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2;
        float ratio = 0.8f;
        std::vector<vector<cv::DMatch>> knnMatches;
        matcher->knnMatch( descSource, descRef, knnMatches, k);

        for (vector<cv::DMatch> matched : knnMatches) {
            if (matched[0].distance < ratio * matched[1].distance)
            {
            matches.push_back(matched[0]);
            } 
        }
    }
    
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor

    double t = (double)cv::getTickCount();
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {   int bytes = 32;
        bool use_orientation = false;
        extractor =  cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int nfeatures = 500;
		float scaleFactor = 1.2f;
		int nlevels = 8;
		int edgeThreshold = 31;
        extractor =  cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientationNormalized=true;
        bool scaleNormalized=true;
        float patternScale=22.0f;
        int nOctaves=4;
        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        int descriptor_size = 0;
		int descriptor_channels = 3;
		float threshold = 0.001f;
		int nOctaves = 4;
		int nOctaveLayers = 4;
        extractor = extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        int nfeatures = 0;
		int nOctaveLayers = 3;
		double contrastThreshold = 0.04;
		double edgeThreshold = 10;
        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold);
    }
    // perform feature description
    extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    double maxOverlap = 0.0;

     // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    for (int row = 0; row < dst_norm.rows; row++) {
        for (int col = 0; col < dst_norm.cols; col++) {

            int response = (int)dst_norm.at<float>(row, col);
            
            if (response<minResponse)
                continue;

            cv::KeyPoint newKeyPoint(cv::Point2f(col, row), 2 * apertureSize, -1, response);
            
            bool placed = false;
            for (auto ptr = keypoints.begin(); ptr != keypoints.end(); ptr++) {
                float overlap = cv::KeyPoint::overlap(newKeyPoint, *ptr);
                if (maxOverlap < overlap) {
                    placed = true;
                    if (newKeyPoint.response > (*ptr).response) {
                        *ptr = newKeyPoint;
                        break;
                    }
                }
            }
            
            if (!placed)
                keypoints.push_back(newKeyPoint);
        }
    }
}

void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    int threshold=30;
    cv::Ptr<cv::FastFeatureDetector> detector=cv::FastFeatureDetector::create(threshold);
    detector->detect(img, keypoints, cv::Mat());

}

void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    int threshold=60;
    int octaves=4;
    float patternScales=1.0f; 

    cv::Ptr<cv::BRISK> detector = cv::BRISK::create(threshold, octaves, patternScales);
    detector->detect(img, keypoints);
}

void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    int nfeatures = 500;

    cv::Ptr<cv::ORB> detector = cv::ORB::create(nfeatures);
    detector->detect(img, keypoints);
}

void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    detector->detect(img, keypoints);
}

void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SiftFeatureDetector::create();
    f2d->detect(img, keypoints);
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) {

    double t = (double)cv::getTickCount();
    if (detectorType.compare("SHITOMASI") == 0)
    {
        detKeypointsShiTomasi(keypoints, img);
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
        detKeypointsHarris(keypoints, img);
    }
    else if (detectorType.compare("FAST") == 0)
    {
        detKeypointsFast(keypoints, img);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detKeypointsBrisk(keypoints, img);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detKeypointsOrb(keypoints, img);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detKeypointsAkaze(keypoints, img);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detKeypointsSift(keypoints, img);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}