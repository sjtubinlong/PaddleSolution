#include <thread>

#include <glog/logging.h>

#include "preprocessor_classify.h"

namespace PaddleSolution {

    bool ClassifyPreProcessor::single_process(const std::string& fname, float* data) {
        // 1. read image
        cv::Mat im = cv::imread(fname);
        if (im.data == nullptr || im.empty()) {
            LOG(ERROR) << "Failed to open image: " << fname;
            return false;
        }
        int channels = im.channels();
        auto ori_w = im.cols;
        auto ori_h = im.rows;
        // 2. resize
        float percent = float(_config->_crop_short_size) / std::min(im.cols, im.rows);
        int rw = round((float)im.cols * percent);
        int rh = round((float)im.rows * percent);
        cv::Size resize_size(rw, rh);
        if (ori_h != rh || ori_w != rw) {
            cv::resize(im, im, resize_size);
        }
        // 3. crop from the center
        int edge = _config->_resize[1];
        int yy = static_cast<int>((im.rows - edge) / 2);
        int xx = static_cast<int>((im.cols - edge) / 2);
        im = cv::Mat(im, cv::Rect(xx, yy, edge, edge));
        // 4. (img - mean) / std
        int hh = im.rows;
        int ww = im.cols;
        int cc = im.channels();
        float* pmean = _config->_mean.data();
        float* pscale = _config->_std.data();
        for (int h = 0; h < hh; ++h) {
            uchar* ptr = im.ptr<uchar>(h);
            int im_index = 0;
            for (int w = 0; w < ww; ++w) {
                for (int c = 0; c < cc; ++c) {
                    int top_index = (c * hh + h) * ww + w;
                    float pixel = static_cast<float>(ptr[im_index++]);
                    pixel = (pixel / 255 - pmean[c]) / pscale[c];
                    data[top_index] = pixel;
                }
            }
        }
        return true;
    }

    bool ClassifyPreProcessor::batch_process(const std::vector<std::string>& imgs, float* data) {
        auto ic = _config->_channels;
        auto iw = _config->_resize[0];
        auto ih = _config->_resize[1];
        std::vector<std::thread> threads;
        for (int i = 0; i < imgs.size(); ++i) {
            std::string path = imgs[i];
            float* buffer = data + i * ic * iw * ih;
            threads.emplace_back([this, path, buffer] {
                single_process(path, buffer);
                });
        }
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        return true;
    }

    bool ClassifyPreProcessor::init(std::shared_ptr<PaddleSolution::PaddleSegModelConfigPaser> config) {
        _config = config;
        return true;
    }
}
