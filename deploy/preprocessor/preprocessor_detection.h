#pragma once

#include "preprocessor.h"

namespace PaddleSolution {

    class DetectionPreProcessor : public ImagePreProcessor {

    public:
        DetectionPreProcessor() : _config(nullptr) {
        };

        bool init(std::shared_ptr<PaddleSolution::PaddleSegModelConfigPaser> config);
         
        bool single_process(const std::string& fname, float* data, int* ori_w, int* ori_h);

        bool batch_process(const std::vector<std::string>& imgs, float* data, int* ori_w, int* ori_h);
    private:
        std::shared_ptr<PaddleSolution::PaddleSegModelConfigPaser> _config;
    };

}
