#include <math.h>
#include <algorithm>
#include <cassert>
#include <vector>

#if defined(NCNN_YOLOX_FLUTTER_IOS)
#include "ncnn/ncnn/net.h"
#else
#include "net.h"
#endif


#if defined(USE_NCNN_SIMPLEOCV)
#if defined(NCNN_YOLOX_FLUTTER_IOS)
#include "ncnn/ncnn/simpleocv.h"
#else
#include "simpleocv.h"
#endif
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

ncnn::Net net;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class TargetBox
{
public:
    float x1;
    float y1;
    float x2;
    float y2;

    int cate;
    float score;

    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };

    float area() { return getWidth() * getHeight(); };
};

class yoloFastestv2 : public ncnn::Layer
{
private:
    std::vector<float> anchor;

    int nmsHandle(std::vector<TargetBox> &tmpBoxes, std::vector<TargetBox> &dstBoxes);
    int getCategory(const float *values, int index, int &category, float &score);
    int predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes,
                   const float scaleW, const float scaleH, const float thresh);

public:
    // ncnn::Net net;

    const char *inputName = "input.1";
    const char *outputName1 = "794"; // 22x22
    const char *outputName2 = "796"; // 11x11

    int numAnchor = 3;
    int numOutput = 1;
    int numThreads = 4;
    int numCategory = 1;
    
    // int numAnchor = 3;
    // int numOutput = 2;
    // int numThreads = 4;
    // int numCategory = 80;

    int inputWidth = 416;
    int inputHeight = 416;

    float nmsThresh = 0.25;

    yoloFastestv2();
    ~yoloFastestv2();

    int loadModel(const char *paramPath, const char *binPath);
    int detection(const cv::Mat srcImg, std::vector<TargetBox> &dstBoxes,
                  const float thresh = 0.3);
};

// 模型的参数配置
yoloFastestv2::yoloFastestv2()
{
    printf("Creat yoloFastestv2 Detector...\n");

    // 打印初始化相关信息
    printf("numThreads:%d\n", numThreads);
    printf("inputWidth:%d inputHeight:%d\n", inputWidth, inputHeight);

    // anchor box w h
    // std::vector<float> bias{12.64, 19.39, 37.88, 51.48, 55.71, 138.31,
    //                         126.91, 78.23, 131.57, 214.55, 279.92, 258.87};

    std::vector<float> bias{213.75, 179.98, 219.65, 335.29, 235.83, 255.04, 276.33, 333.70, 308.07, 245.34, 338.17, 337.73};

    anchor.assign(bias.begin(), bias.end());
}

yoloFastestv2::~yoloFastestv2()
{
    printf("Destroy yoloFastestv2 Detector...\n");
}

// ncnn 模型加载
int yoloFastestv2::loadModel(const char *paramPath, const char *binPath)
{
    printf("Ncnn mode init:\n%s\n%s\n", paramPath, binPath);
    NCNN_LOGE("Ncnn mode init:\n%s\n%s\n", paramPath, binPath);

    net.load_param(paramPath);
    net.load_model(binPath);

    NCNN_LOGE("Ncnn model init sucess...\n");

    return 0;
}

float intersection_area(const TargetBox &a, const TargetBox &b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b)
{
    return (a.score > b.score);
}

// NMS处理
int yoloFastestv2::nmsHandle(std::vector<TargetBox> &tmpBoxes,
                             std::vector<TargetBox> &dstBoxes)
{
    std::vector<int> picked;

    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for (int i = 0; i < tmpBoxes.size(); i++)
    {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++)
        {
            // 交集
            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            // 并集
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if (IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate)
            {
                keep = 0;
                break;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++)
    {
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;
}

// 检测类别分数处理
int yoloFastestv2::getCategory(const float *values, int index, int &category, float &score)
{
    float tmp = 0;
    float objScore = values[4 * numAnchor + index];

    for (int i = 0; i < numCategory; i++)
    {
        float clsScore = values[4 * numAnchor + numAnchor + i];
        clsScore *= objScore;

        if (clsScore > tmp)
        {
            score = clsScore;
            category = i;

            tmp = clsScore;
        }
    }

    return 0;
}

// 特征图后处理
int yoloFastestv2::predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes,
                              const float scaleW, const float scaleH, const float thresh)
{ // do result
    for (int i = 0; i < numOutput; i++)
    {
        int stride;
        int outW, outH, outC;

        outH = out[i].c;
        outW = out[i].h;
        outC = out[i].w;

        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        for (int h = 0; h < outH; h++)
        {
            const float *values = out[i].channel(h);

            for (int w = 0; w < outW; w++)
            {
                for (int b = 0; b < numAnchor; b++)
                {
                    // float objScore = values[4 * numAnchor + b];
                    TargetBox tmpBox;
                    int category = -1;
                    float score = -1;

                    getCategory(values, b, category, score);

                    if (score > thresh)
                    {
                        float bcx, bcy, bw, bh;

                        bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
                        bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
                        bw = pow((values[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                        bh = pow((values[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];

                        float x1 = abs((bcx - 0.5 * bw) * scaleW);
                        float y1 = (bcy - 0.5 * bh) * scaleH;
                        float x2 = (bcx + 0.5 * bw) * scaleW;
                        float y2 = (bcy + 0.5 * bh) * scaleH;


                        tmpBox.x1 = x1;
                        tmpBox.y1 = y1;
                        tmpBox.x2 = x2;
                        tmpBox.y2 = y2;
                        tmpBox.score = score;
                        tmpBox.cate = category;

                        NCNN_LOGE("x1:%f,y1:%f,x2:%f,y2:%f,score:%f,cate:%d", tmpBox.x1, tmpBox.y1, tmpBox.x2, tmpBox.y2, tmpBox.score, tmpBox.cate);

                        // tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                        // tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                        // tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                        // tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                        // tmpBox.score = score;
                        // tmpBox.cate = category;

                        dstBoxes.push_back(tmpBox);
                    }
                }
                values += outC;
            }
        }
    }
    return 0;
}

int yoloFastestv2::detection(const cv::Mat srcImg, std::vector<TargetBox> &dstBoxes, const float thresh)
{
    dstBoxes.clear();

    float scaleW = (float)srcImg.cols / (float)inputWidth;
    float scaleH = (float)srcImg.rows / (float)inputHeight;

    // resize of input image data
    ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR,
                                                       srcImg.cols, srcImg.rows, inputWidth, inputHeight);

    // Normalization of input image data
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    inputImg.substract_mean_normalize(mean_vals, norm_vals);

    // creat extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(numThreads);

    // set input tensor
    ex.input(inputName, inputImg);
    NCNN_LOGE("masuk++");

    // forward
    ncnn::Mat out[2];
    ex.extract(outputName1, out[0]); // 22x22
    ex.extract(outputName2, out[1]); // 11x11

    std::vector<TargetBox> tmpBoxes;
    // 特征图后处理
    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);

    // NMS
    nmsHandle(tmpBoxes, dstBoxes);

    return 0;
}

DEFINE_LAYER_CREATOR(yoloFastestv2)

extern "C" __attribute__((visibility("default"))) __attribute__((used)) void initYolox(char *modelPath, char *paramPath)
{

    yoloFastestv2 yfastest;
    net.register_custom_layer("yoloFastestv2", yoloFastestv2_layer_creator);
    net.load_param(paramPath);
    net.load_model(modelPath);

    // yfastest.loadModel(paramPath, modelPath);
    
}

int drawBoxes(cv::Mat srcImg, std::vector<TargetBox> boxes, std::vector<Object> output)
{
    for (int i = 0; i < boxes.size(); i++)
    {
        // Assign rectangle coordinates to rect in Object
        Object obj;
        obj.rect.x = boxes[i].x1;
        obj.rect.y = boxes[i].y1;
        obj.rect.width = boxes[i].getWidth();
        obj.rect.height = boxes[i].getHeight();
        obj.label = boxes[i].cate;
        obj.prob = boxes[i].score;
        output.push_back(obj);
    }

    return 0;
}

char *parseResultsObjects(std::vector<TargetBox> boxes)
{
    if (boxes.size() == 0) 
    {
        NCNN_LOGE("No object detected");
        return (char *)"";
    }

    NCNN_LOGE("Number of objects detected: %zu", boxes.size());

    std::string result = "";
    for (int i = 0; i < (int)boxes.size(); i++)
    {
        // NCNN_LOGE("Object %zu: %zu, %zu, %zu, %zu, %zu, %zu", i, boxes[i].x1, boxes[i].y1, boxes[i].getWidth(), boxes[i].getHeight(), boxes[i].cate, boxes[i].score);
        NCNN_LOGE("=============++=============");

        TargetBox obj = boxes[i];
        NCNN_LOGE("=============+a+=============");
        result += std::to_string(obj.x1) + "," + std::to_string(obj.y1) + "," + std::to_string(obj.getWidth()) + "," + std::to_string(obj.getHeight()) + "," + std::to_string(obj.cate) + "," + std::to_string(obj.score);
        NCNN_LOGE("=============+ab+=============");
        // NCNN_LOGE("Object %d: %d, %d, %d, %d, %d, %d", i, obj.x1, obj.y1, obj.getWidth(), obj.getHeight(), obj.cate, obj.score);
    }


    char *result_c = new char[result.length() + 1];
        NCNN_LOGE("=============+abc+=============");
    strcpy(result_c, result.c_str());
        NCNN_LOGE("=============+abcd+=============");
    NCNN_LOGE("Result: %s", result_c);

    return result_c;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used)) char *detectWithImagePath(char *imagepath, double nms_thresh, double conf_thresh, int target_size)
{
    // Load image using OpenCV
    cv::Mat m = cv::imread(imagepath, cv::IMREAD_COLOR);

    // Exit if the image does not load.
    if (m.empty())
    {
        NCNN_LOGE("cv::imread %s failed", imagepath);
        return (char *)"";
    }

    std::vector<TargetBox> objects;
    yoloFastestv2 yolo;

    // Measure execution time
    auto start_time = std::chrono::steady_clock::now();

    yolo.detection(m, objects, (float)conf_thresh);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    NCNN_LOGE("Processing time: %ld milliseconds", duration_ms);

    // drawBoxes(m, objects, objects2);

    return parseResultsObjects(objects);
}

cv::Mat convertToMat(const unsigned char *pixels, int width, int height)
{
    cv::Mat image(height, width, CV_8UC1);
    memcpy(image.data, pixels, width * height);

    return image;
}


extern "C" __attribute__((visibility("default"))) __attribute__((used)) void disposeYolox()
{
    net.clear();
}

extern "C" __attribute__((visibility("default"))) __attribute__((used)) char *detectWithPixels(const unsigned char *pixels, int pixelType, int width, int height, double nms_thresh, double conf_thresh, int target_size)
{
    yoloFastestv2 yolo;

    int w = yolo.inputWidth;
    int h = yolo.inputHeight;
    ncnn::Mat image = ncnn::Mat::from_pixels_resize(pixels, pixelType, yolo.inputWidth, yolo.inputHeight, w, h);
    std::vector<TargetBox> objects;

    // Convert ncnn::Mat to cv::Mat
    cv::Mat cvImage(w, h, CV_8UC3, image.data);

    // Measure execution time
    auto start_time = std::chrono::steady_clock::now();

    yolo.detection(cvImage, objects, (float)conf_thresh);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    NCNN_LOGE("Processing time: %ld milliseconds", duration_ms);

    NCNN_LOGE("masuk sini");

    return parseResultsObjects(objects);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used)) void yuv420sp2rgb(const unsigned char *yuv420sp, int width, int height, unsigned char *rgb)
{
    ncnn::yuv420sp2rgb(yuv420sp, width, height, rgb);
    return;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used)) void rgb2rgba(const unsigned char *rgb, int width, int height, unsigned char *rgba)
{
    ncnn::Mat m = ncnn::Mat::from_pixels(rgb, ncnn::Mat::PIXEL_RGB2BGRA, width, height);
    m.to_pixels(rgba, ncnn::Mat::PIXEL_RGBA);
    return;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used)) void kannaRotate(const unsigned char *src, int channel, int srcw, int srch, unsigned char *dst, int dsw, int dsh, int type)
{
    switch (channel)
    {
    case 1:
        ncnn::kanna_rotate_c1(src, srcw, srch, dst, dsw, dsh, type);
        break;
    case 2:
        ncnn::kanna_rotate_c2(src, srcw, srch, dst, dsw, dsh, type);
        break;
    case 3:
        ncnn::kanna_rotate_c3(src, srcw, srch, dst, dsw, dsh, type);
        break;
    case 4:
        ncnn::kanna_rotate_c4(src, srcw, srch, dst, dsw, dsh, type);
        break;
    }
    return;
}
