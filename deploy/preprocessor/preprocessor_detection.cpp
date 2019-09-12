#include <thread>
#include <mutex>

#include <glog/logging.h>

#include "preprocessor_detection.h"
#include "utils/utils.h"

namespace PaddleSolution {
    std::mutex gMutex;
    bool DetectionPreProcessor::single_process(const std::string& fname, std::vector<float> &vec_data, int* ori_w, int* ori_h, int* resize_w, int* resize_h, float* scale_ratio) {
        cv::Mat im1 = cv::imread(fname, -1);
	cv::Mat im;
	if(_config->_feeds_size == 3) { // faster rcnn
            im1.convertTo(im, CV_32FC3, 1/255.0);
        }
	else if(_config->_feeds_size == 2){ //yolo v3
	    im = im1;
        }
        if (im.data == nullptr || im.empty()) {
            LOG(ERROR) << "Failed to open image: " << fname;
            return false;
        }
        
        int channels = im.channels();
        *ori_w = im.cols;
        *ori_h = im.rows;
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);      
        channels = im.channels();
        //print whole image
//	for(int h = 0; h < im.rows; ++h) {
//	    const uchar* ptr = im.ptr<uchar>(h);
//	    for(int w = 0; w < im.cols; ++w) {
//		for(int c = 0; c < channels; ++c) {
//		    std::cout << (int)ptr[w * channels + c] << " ";
//		}
//		std::cout << std::endl;
//	    } 
//            std::cout << std::endl;
//	}


	//resize
        int rw = im.cols;
        int rh = im.rows;
	float im_scale_ratio;
	std::cout << im.cols << "," << im.rows << std::endl;
	utils::scaling(_config->_resize_type, rw, rh, _config->_resize[0], _config->_resize[1], _config->_target_short_size, _config->_resize_max_size, im_scale_ratio);
        cv::Size resize_size(rw, rh);
	*resize_w = rw;
	*resize_h = rh;
	*scale_ratio = im_scale_ratio;
        if (*ori_h != rh || *ori_w != rw) {
	    cv::Mat im_temp;
	    if(_config->_resize_type == utils::SCALE_TYPE::UNPADDING) {
                cv::resize(im, im_temp, resize_size, 0, 0, cv::INTER_LINEAR);
	    }
	    else if(_config->_resize_type == utils::SCALE_TYPE::RANGE_SCALING) {
            	cv::resize(im, im_temp, cv::Size(), im_scale_ratio, im_scale_ratio, cv::INTER_LINEAR);
	    }
	    im = im_temp;
        }

	vec_data.resize(channels * rw * rh);
//	std::cout << channels << " " << im.cols << " " << im.rows << " " << im_scale_ratio << std::endl;
//        printf("%d %d %10.7f\n", im.cols, im.rows, im_scale_ratio);
        float *data = vec_data.data();

        float* pmean = _config->_mean.data();
        float* pscale = _config->_std.data();
//	for(int i = 0; i <  channels; ++i){
//	    std::cout << "mean = " << pmean[i] << " std = " << pscale[i] << std::endl;
//	}
        for (int h = 0; h < rh; ++h) {
            const uchar* uptr = im.ptr<uchar>(h);
            const float* fptr = im.ptr<float>(h);
            int im_index = 0;
            for (int w = 0; w < rw; ++w) {
                for (int c = 0; c < channels; ++c) {
                    int top_index = (c * rh + h) * rw + w;
                    float pixel;// = static_cast<float>(fptr[im_index]);// / 255.0;
	            if(_config->_feeds_size == 2){ //yolo v3
			pixel = static_cast<float>(uptr[im_index++]) / 255.0;
                    }
		    else if(_config->_feeds_size == 3){
                        pixel = fptr[im_index++];
		    }
                    pixel = (pixel - pmean[c]) / pscale[c];
                    data[top_index] = pixel;
                }
            }
        }
//	const uchar* ptr = im.ptr<uchar>(0);
//        const float* ptr = im.ptr<float>(0);
//	for(int i = 0; i < rw; ++i) {
//	    std::cout << (int)((data[i]*pscale[0] + pmean[0])*255) << " ";
//	    std::cout << (int)ptr[i * channels] << " ";
//	    std::cout << ptr[i * channels] << " ";
//	    std::cout << data[i] << " ";
//	    if((i + 1) % 40 == 0){
//		std::cout << std::endl;
//            }
//	}
//	std::cout << std::endl;
        return true;
    }

    bool DetectionPreProcessor::batch_process(const std::vector<std::string>& imgs, std::vector<float> &data, int* ori_w, int* ori_h, int* resize_w, int* resize_h, float* scale_ratio) {
        auto ic = _config->_channels;
        auto iw = _config->_resize[0];
        auto ih = _config->_resize[1];
        std::vector<std::thread> threads;
        for (int i = 0; i < imgs.size(); ++i) {
            std::string path = imgs[i];
            int* width = &ori_w[i];
            int* height = &ori_h[i];
	    int* resize_width = &resize_w[i];
	    int* resize_height = &resize_h[i];
	    float* sr = &scale_ratio[i];
            threads.emplace_back([this, &data, path, width, height, resize_width, resize_height, sr] {
                std::vector<float> buffer;
                single_process(path, buffer, width, height, resize_width, resize_height, sr);
		gMutex.lock();
		data.insert(data.end(), buffer.begin(), buffer.end());
		gMutex.unlock();
                });
        }
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        return true;
    }

    bool DetectionPreProcessor::init(std::shared_ptr<PaddleSolution::PaddleSegModelConfigPaser> config) {
        _config = config;
        return true;
    }

}
