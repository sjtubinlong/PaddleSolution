#include "classify_predictor.h"

namespace PaddleSolution {

    int ClassifyPredictor::init(const std::string& conf) {
        if (!_model_config.load_config(conf)) {
            LOG(FATAL) << "Fail to load config file: [" << conf << "]";
            return -1;
        }
        _preprocessor = PaddleSolution::create_processor(conf);
        if (_preprocessor == nullptr) {
            LOG(FATAL) << "Failed to create_processor";
            return -1;
        }

        bool use_gpu = _model_config._use_gpu;
        const auto& model_dir = _model_config._model_path;
        const auto& model_filename = _model_config._model_file_name;
        const auto& params_filename = _model_config._param_file_name;

        // load paddle model file
        if (_model_config._predictor_mode == "NATIVE") {
            paddle::NativeConfig config;
            auto prog_file = utils::path_join(model_dir, model_filename);
            auto param_file = utils::path_join(model_dir, params_filename);
            config.prog_file = prog_file;
            config.param_file = param_file;
            config.fraction_of_gpu_memory = 0;
            config.use_gpu = use_gpu;
            config.device = 0;
            _main_predictor = paddle::CreatePaddlePredictor(config);
        } else if (_model_config._predictor_mode == "ANALYSIS") {
            paddle::AnalysisConfig config;
            if (use_gpu) {
                config.EnableUseGpu(100, 0);
            }
            auto prog_file = utils::path_join(model_dir, model_filename);
            auto param_file = utils::path_join(model_dir, params_filename);
            config.SetModel(prog_file, param_file);
            config.SwitchUseFeedFetchOps(false);
            _main_predictor = paddle::CreatePaddlePredictor(config);
        } else {
            return -1;
        }
        return 0;

    }

    int ClassifyPredictor::predict(const std::vector<std::string>& imgs) {
        if (_model_config._predictor_mode == "NATIVE") {
            return native_predict(imgs);
        }
        else if (_model_config._predictor_mode == "ANALYSIS") {
            return analysis_predict(imgs);
        }
        return -1;
    }

    int ClassifyPredictor::native_predict(const std::vector<std::string>& imgs)
    {
        int config_batch_size = _model_config._batch_size;

        int channels = _model_config._channels;
        int eval_width = _model_config._resize[0];
        int eval_height = _model_config._resize[1];
        std::size_t total_size = imgs.size();
        int default_batch_size = std::min(config_batch_size, (int)total_size);
        int batch = total_size / default_batch_size + ((total_size % default_batch_size) != 0);
        int batch_buffer_size = default_batch_size * channels * eval_width * eval_height;

        auto& input_buffer = _buffer;
        auto& imgs_batch = _imgs_batch;

        input_buffer.resize(batch_buffer_size);
        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels * eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            input_buffer.resize(real_buffer_size);
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }
            if (!_preprocessor->batch_process(imgs_batch, input_buffer.data())) {
                return -1;
            }

            paddle::PaddleTensor im_tensor;
            im_tensor.name = "image";
            im_tensor.shape = std::vector<int>({ batch_size, channels, eval_height, eval_width });
            im_tensor.data.Reset(input_buffer.data(), real_buffer_size * sizeof(float));
            im_tensor.dtype = paddle::PaddleDType::FLOAT32;
            feeds.push_back(im_tensor);
            _outputs.clear();
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!_main_predictor->Run(feeds, &_outputs, batch_size)) {
                LOG(ERROR) << "Failed: NativePredictor->Run() return false at batch: " << u;
                continue;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            std::cout << "runtime = " << duration << std::endl;
            int out_num = 1;
            // print shape of first output tensor for debugging
            std::cout << "size of outputs[" << 0 << "]: (";
            for (int j = 0; j < _outputs[0].shape.size(); ++j) {
                out_num *= _outputs[0].shape[j];
                std::cout << _outputs[0].shape[j] << ",";
            }
            std::cout << ")" << std::endl;
            const size_t nums = _outputs.front().data.length() / sizeof(float);
            if (out_num % batch_size != 0 || out_num != nums) {
                LOG(ERROR) << "outputs data size mismatch with shape size.";
                return -1;
            }

            for (int i = 0; i < batch_size; ++i) {
                float* out_addr = (float*)(_outputs[0].data.data()) + i * (out_num / batch_size);
            int max_idx = 0;
                for (int j = 0; j < (out_num / batch_size); ++j) {
                if(*(j + out_addr) > *(max_idx + out_addr)){
                  max_idx = j;
                }
                    printf("img[%s], class[%d], score = [%e]\n", imgs_batch[i].c_str(), j, *(j + out_addr));
                }
                std::cout << "class: " << max_idx << "\tscore:" << *(max_idx + out_addr) << std::endl;
            }
        }

        return 0;
    }

    int ClassifyPredictor::analysis_predict(const std::vector<std::string>& imgs) {

        int config_batch_size = _model_config._batch_size;
        int channels = _model_config._channels;
        int eval_width = _model_config._resize[0];
        int eval_height = _model_config._resize[1];
        auto total_size = imgs.size();
        int default_batch_size = std::min(config_batch_size, (int)total_size);
        int batch = total_size / default_batch_size + ((total_size % default_batch_size) != 0);
        int batch_buffer_size = default_batch_size * channels * eval_width * eval_height;

        auto& input_buffer = _buffer;
        auto& imgs_batch = _imgs_batch;
        input_buffer.resize(batch_buffer_size);

        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels * eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            input_buffer.resize(real_buffer_size);
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }
            if (!_preprocessor->batch_process(imgs_batch, input_buffer.data())) {
                return -1;
            }
            auto im_tensor = _main_predictor->GetInputTensor("image");
            im_tensor->Reshape({ batch_size, channels, eval_height, eval_width });
            im_tensor->copy_from_cpu(input_buffer.data());

            auto t1 = std::chrono::high_resolution_clock::now();
            _main_predictor->ZeroCopyRun();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            std::cout << "runtime = " << duration << std::endl;

            auto output_names = _main_predictor->GetOutputNames();
            auto output_t = _main_predictor->GetOutputTensor(output_names[0]);
            std::vector<float> out_data;
            std::vector<int> output_shape = output_t->shape();

            int out_num = 1;
            std::cout << "size of outputs[" << 0 << "]: (";
            for (int j = 0; j < output_shape.size(); ++j) {
                out_num *= output_shape[j];
                std::cout << output_shape[j] << ",";
            }
            std::cout << ")" << std::endl;

            out_data.resize(out_num);
            output_t->copy_to_cpu(out_data.data());
            for (int i = 0; i < batch_size; ++i) {
                float* out_addr = out_data.data() + (out_num / batch_size) * i;
                int max_idx = 0;
                for (int j = 0; j < (out_num / batch_size); ++j) {
                    printf("img[%s], class[%d], score = [%e]\n", imgs_batch[i].c_str(), j, *(j + out_addr));
                    if(*(j + out_addr) > *(max_idx + out_addr)) {
                        max_idx = j;
                    }
                }
                std::cout << "class: " << max_idx << "\tscore:" << *(max_idx + out_addr) << std::endl;
            }
        }
        return 0;
    }
}
