//
// Created by amey on 05.07.21.
//

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

// Include the CPU Implementation
#include "CPU_impl/cpu_impl.h"

///////////// Declarations

//std::string input_image_path = "../Images/Inputs/Binary_Image/image_1.pgm";
//std::string input_image_path = "../Images/Valve.pgm";
//std::string output_path = "../Images/Outputs/";


///////////// Local Functions/ Methods

/**
 * getCountSize() :         Returns all the countX, countY, count and size based on image width and height
 * */

auto getCount_Size(
        std::size_t wgSizeX,
        std::size_t wgSizeY,
        std::size_t inWidth,
        std::size_t inHeight
        ){
    std::size_t cntX, cntY, cnt, sze;
    cntX = wgSizeX * std::size_t(inWidth / wgSizeX);
    cntY = wgSizeY * std::size_t(inHeight / wgSizeY);
    cnt = cntX * cntY;
    sze = cnt * sizeof(float);

    struct result {std::size_t countX; std::size_t countY; std::size_t count; std::size_t size;};
    return result {cntX, cntY, cnt, sze};

}

cl::Program buildProgram(cl::Context context, std::vector<cl::Device> devices, std::string filepath){
    cl::Program program = OpenCL::loadProgramSource(context, filepath);
    OpenCL::buildProgram(program, devices);
    return program;
}


int main(int argc, char **argv) {
    /**
     * Define the command line arguments here
     * */

    std::string input_image_path = argv[1];
    std::string output_path = argv[2];


    // Create a context
    //cl::Context context(CL_DEVICE_TYPE_GPU);
    std::vector <cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cerr << "No platforms found" << std::endl;
        return 1;
    }
    int platformId = 0;
    for (size_t i = 0; i < platforms.size(); i++) {
        if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
            platformId = i;
            break;
        }
    }
    cl_context_properties prop[4] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId](), 0, 0};
    std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '"
              << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
    cl::Context context(CL_DEVICE_TYPE_GPU, prop);

    // Get the first device of the context
    std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
    std::vector <cl::Device> devices;
    devices.push_back(device);
    OpenCL::printDeviceInfo(std::cout, device);
    std::cout << "Number of Cores: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Number of WorkGroups (max): " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
    std::cout << "Number of WorkItem Dimensions (max): " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;

    // Create a command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Declare some values
    std::size_t wgSizeX = 32;                          // Number of work items per work group in X direction
    std::size_t wgSizeY = 32;
//    std::size_t countX ;                               // Overall number of work items in X direction = Number of elements in X direction
//    std::size_t countY ;                               // countX *= 3; countY *= 3;
//    std::size_t count ;                                // Overall number of elements
//    std::size_t size ;                                 // Size of data in bytes

    /**
     * Read the test image here
     * */
    std::vector<float> inputData;
    std::size_t inputWidth, inputHeight;
    Core::readImagePGM(input_image_path, inputData, inputWidth, inputHeight);
    auto[countX, countY, count, size] = getCount_Size(
            wgSizeX,
            wgSizeY,
            inputWidth,
            inputHeight
    );


    // Allocate space for output data from CPU and GPU on the host
    std::vector<float> h_input(count);
    std::vector<float> h_outputCpu(count);
    std::vector<float> h_outputGpu(count);

    // Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
    memset(h_input.data(), 255, size);
    memset(h_outputCpu.data(), 255, size);
    memset(h_outputGpu.data(), 255, size);

    {

        for (size_t j = 0; j < countY; j++) {
            for (size_t i = 0; i < countX; i++) {
                h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
            }
        }
    }

    // Do calculation on the host side
    Core::TimeSpan t1 = Core::getCurrentTime();
    median_filter(h_input, 5, h_outputCpu, countX, countY);
//    add_salt_and_pepper(h_input, h_outputCpu, countX, countY);
    Core::TimeSpan cpuTime = Core::getCurrentTime() - t1;

    //////// Store CPU output image ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_sobel_cpu.pgm", h_outputCpu, countX, countY);


    //////// Load Images in the device ////////////////////////////////

    auto d_input_image = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
    auto d_output_image = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);

    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = countX;
    region[1] = countY;
    region[2] = 1;

    cl::Event imageUpload;                              // image upload event

    queue.enqueueWriteImage(
            d_input_image,
            true,
            origin,
            region,
            countX * sizeof(float),
            0,
            h_input.data(),
            NULL,
            NULL
    );


    /////////// Load Buffers ///////////////////
    cl::Buffer d_input_buffer(context, CL_MEM_READ_WRITE, size);
    cl::Buffer d_output_buffer(context, CL_MEM_READ_WRITE, size);

    queue.enqueueWriteBuffer(d_input_buffer, true, 0, size, h_input.data());


    /**
     * Load All the CL programs that you design here:
     * */

    cl::Program median_program = buildProgram(context, devices, "../src/median.cl");
    cl::Kernel median_kernel_3(median_program, "medianKernel_5");
    cl::Kernel median_kernel_image_3(median_program, "medianKernel_image_5");


    /////////////////// Run the Kernel ///////////////////////


    ////////////////// Median Kernel : Bufferized ////////////
    cl::Event median_buffer_execution;
    median_kernel_3.setArg<cl::Buffer>(0, d_input_buffer);
    median_kernel_3.setArg<cl::Buffer>(1, d_output_buffer);

    queue.enqueueNDRangeKernel(median_kernel_3,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &median_buffer_execution);

    queue.enqueueReadBuffer(d_output_buffer, true, 0, size, h_outputGpu.data(), NULL, NULL);

    //////// Store GPU output image : Bufferized Median Kernel ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_sobel_gpu_bufferized.pgm", h_outputGpu, countX, countY);

    std::cout << "INFO:\t MEDIAN FILTER (Bufferized):\tSpeedup = "<< (cpuTime.getSeconds() / OpenCL::getElapsedTime(median_buffer_execution).getSeconds()) << std::endl;


    //////// Median Kernel : with Image2D //////////////
    cl::Event median_image_execution;
    median_kernel_image_3.setArg<cl::Image2D>(0, d_input_image);
    median_kernel_image_3.setArg<cl::Image2D>(1, d_output_image);

    queue.enqueueNDRangeKernel(median_kernel_image_3,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &median_image_execution);

    queue.enqueueReadImage(d_output_image,
                           true,
                           origin,
                           region,
                           countX * sizeof(float),
                           0,
                           h_outputGpu.data(),
                           nullptr,
                           NULL
    );
    //////// Store GPU output image : Image Median Kernel ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_sobel_gpu_image.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t MEDIAN FILTER (Image2D):\tSpeedup = "<< (cpuTime.getSeconds() / OpenCL::getElapsedTime(median_image_execution).getSeconds()) << std::endl;


}
