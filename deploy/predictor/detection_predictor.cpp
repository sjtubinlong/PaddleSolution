#include "detection_predictor.h"

namespace PaddleSolution {

    int DetectionPredictor::init(const std::string& conf) {
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

    int DetectionPredictor::predict(const std::vector<std::string>& imgs) {
        if (_model_config._predictor_mode == "NATIVE") {
            return native_predict(imgs);
        }
        else if (_model_config._predictor_mode == "ANALYSIS") {
            return analysis_predict(imgs);
        }
        return -1;
    }

    int DetectionPredictor::native_predict(const std::vector<std::string>& imgs) {
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
	float sr;
        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels * eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            input_buffer.clear();
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }
	    std::vector<int> ori_widths;
	    std::vector<int> ori_heights;
	    std::vector<int> resize_widths;
	    std::vector<int> resize_heights;
            std::vector<float> scale_ratios;
            ori_widths.resize(batch_size);
            ori_heights.resize(batch_size);
	    resize_widths.resize(batch_size);
	    resize_heights.resize(batch_size);
	    scale_ratios.resize(batch_size);            

            if (!_preprocessor->batch_process(imgs_batch, input_buffer, ori_widths.data(), ori_heights.data(),
					      resize_widths.data(), resize_heights.data(), scale_ratios.data())) {
                return -1;
            }
            paddle::PaddleTensor im_tensor, im_size_tensor, im_info_tensor;

            im_tensor.name = "image";
            im_tensor.shape = std::vector<int>({ batch_size, channels, resize_heights[0], resize_widths[0] });
            im_tensor.data.Reset(input_buffer.data(), input_buffer.size() * sizeof(float));
            im_tensor.dtype = paddle::PaddleDType::FLOAT32;
 
	    std::vector<float> image_infos;
            for(int i = 0; i < batch_size; ++i) {
		image_infos.push_back(resize_heights[i]);
		image_infos.push_back(resize_widths[i]);
		image_infos.push_back(scale_ratios[i]);
	    }
	    im_info_tensor.name = "info";
	    im_info_tensor.shape = std::vector<int>({batch_size, 3});
	    im_info_tensor.data.Reset(image_infos.data(), batch_size * 3 * sizeof(float));
	    im_info_tensor.dtype = paddle::PaddleDType::FLOAT32;

	    std::vector<int> image_size;
	    for(int i = 0; i < batch_size; ++i) {
		image_size.push_back(ori_heights[i]);
		image_size.push_back(ori_widths[i]);
	    }

	    std::vector<float> image_size_f;
	    for(int i = 0; i < batch_size; ++i) {
		image_size_f.push_back(ori_heights[i]);
		image_size_f.push_back(ori_widths[i]);
		image_size_f.push_back(1.0);
	    }

	    int feeds_size = _model_config._feeds_size;
	    im_size_tensor.name = "im_size";
	    if(feeds_size == 2) {
                im_size_tensor.shape = std::vector<int>({ batch_size, 2});
                im_size_tensor.data.Reset(image_size.data(), batch_size * 2 * sizeof(int));
	        im_size_tensor.dtype = paddle::PaddleDType::INT32;
            }
	    else if(feeds_size == 3) {
                im_size_tensor.shape = std::vector<int>({ batch_size, 3});
                im_size_tensor.data.Reset(image_size_f.data(), batch_size * 3 * sizeof(float));
	        im_size_tensor.dtype = paddle::PaddleDType::FLOAT32;
            }
	    std::cout << "Feed size = " << feeds_size << std::endl;
            feeds.push_back(im_tensor);
	    if(_model_config._feeds_size > 2) {
	        feeds.push_back(im_info_tensor);
            }
            feeds.push_back(im_size_tensor);
            _outputs.clear();

            auto t1 = std::chrono::high_resolution_clock::now();
            if (!_main_predictor->Run(feeds, &_outputs, batch_size)) {
                LOG(ERROR) << "Failed: NativePredictor->Run() return false at batch: " << u;
                continue;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            std::cout << "runtime = " << duration << std::endl;
            std::cout << "Number of outputs:"  << _outputs.size() << std::endl;
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
	    float* out_addr = (float *)(_outputs[0].data.data());
	    for(int i = 0; i < _outputs[0].lod[0].size() - 1; ++i){
               for (int j = _outputs[0].lod[0][i]; j < _outputs[0].lod[0][i+1]; ++j) {
		  printf("Class %d, score = %f, left top = [%f, %f], right bottom = [%f, %f]\n",
	        	 static_cast<int>(round(out_addr[0 + j * 6])), out_addr[1 + j * 6], out_addr[2 + j * 6], 
                                                out_addr[3 + j * 6], out_addr[4 + j * 6], out_addr[5 + j * 6]);	
		}
		printf("\n");
	    }
        }

        return 0;
    }

    int DetectionPredictor::analysis_predict(const std::vector<std::string>& imgs) {

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

        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels * eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            //input_buffer.resize(real_buffer_size);
	    input_buffer.clear();
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }
	    
            std::vector<int> ori_widths;
	    std::vector<int> ori_heights;
	    std::vector<int> resize_widths;
	    std::vector<int> resize_heights;
            std::vector<float> scale_ratios;
            ori_widths.resize(batch_size);
            ori_heights.resize(batch_size);
	    resize_widths.resize(batch_size);
	    resize_heights.resize(batch_size);
	    scale_ratios.resize(batch_size);

            if (!_preprocessor->batch_process(imgs_batch, input_buffer, ori_widths.data(), ori_heights.data(),
					      resize_widths.data(), resize_heights.data(), scale_ratios.data())){
		std::cout << "Failed to preprocess!" << std::endl;
                return -1;
            }
	    std::vector<std::string> input_names = _main_predictor->GetInputNames();
//	    for(auto name : input_names){
//		std::cout << name << std::endl;
//	    }
//	    std::cout << "Input buffer size = " << input_buffer.size() << std::endl;
            auto im_tensor = _main_predictor->GetInputTensor(input_names.front());
            im_tensor->Reshape({ batch_size, channels, resize_heights[0], resize_widths[0]});
            im_tensor->copy_from_cpu(input_buffer.data());
 
//           for(int i = 0; i < resize_widths[0]; ++i) {
//		std::cout << input_buffer[i]  << " ";
//	    } 
//            std::cout << std::endl;
	    if(input_names.size() > 2){
     	        std::vector<float> image_infos;
                for(int i = 0; i < batch_size; ++i) {
		    image_infos.push_back(resize_heights[i]);
		    image_infos.push_back(resize_widths[i]);
		    image_infos.push_back(scale_ratios[i]);
		    std::cout << resize_heights[i] << " " << resize_widths[i] << " " << scale_ratios[i] << std::endl;
	        }		
	    	auto im_info_tensor = _main_predictor->GetInputTensor(input_names[1]);
	        im_info_tensor->Reshape({batch_size, 3});
	        im_info_tensor->copy_from_cpu(image_infos.data());
	    }

	    std::vector<int> image_size;
	    for(int i = 0; i < batch_size; ++i) {
		image_size.push_back(ori_heights[i]);
		image_size.push_back(ori_widths[i]);
	    }
            std::vector<float> image_size_f;
	    for(int i = 0; i < batch_size; ++i) {
		image_size_f.push_back(static_cast<float>(ori_heights[i]));
		image_size_f.push_back(static_cast<float>(ori_widths[i]));
		image_size_f.push_back(1.0);
		
	//	std::cout << ori_heights[i] << " " << ori_widths[i] << " 1.0" << std::endl;
	    }
             
            auto im_size_tensor = _main_predictor->GetInputTensor(input_names.back());
	    if(input_names.size() > 2) {
	        im_size_tensor->Reshape({batch_size, 3});
	    	im_size_tensor->copy_from_cpu(image_size_f.data());
	    }
	    else{
	        im_size_tensor->Reshape({batch_size, 2});
	    	im_size_tensor->copy_from_cpu(image_size.data());
	    }
	    

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

	    float* out_addr = (float *)(out_data.data());
            auto lod_vector = output_t->lod();
	    for(int i = 0; i < lod_vector[0].size() - 1; ++i){
               for (int j = lod_vector[0][i]; j < lod_vector[0][i+1]; ++j) {
		  printf("Class %d, score = %f, left top = [%f, %f], right bottom = [%f, %f]\n",
	        	 static_cast<int>(round(out_addr[0 + j * 6])), out_addr[1 + j * 6], out_addr[2 + j * 6], 
                                                out_addr[3 + j * 6], out_addr[4 + j * 6], out_addr[5 + j * 6]);	
		}
		printf("\n");
	    }
        }
        return 0;
    }
}