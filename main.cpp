#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include "imageBuffer.h"
#include <chrono>
#include <thread>

using namespace std;

class MobileNetSSD {
    public:
        MobileNetSSD();
        ~MobileNetSSD();
        void init();
        float* allocateMemory(DimsCHW dims, char* info);
        cv::Mat load_matrix_from_python(int * matrix, int rows, int cols, int channels);
        void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale );
        int detect(int * matrix, int rows, int cols, int channels); 
        void destory();
        float* getOutput();
    private:
        const char* INPUT_BLOB_NAME;
        const char* OUTPUT_BLOB_NAME;
        static const uint32_t BATCH_SIZE = 1;
        ConsumerProducerQueue<cv::Mat> *imageBuffer;
        std::vector<std::string> output_vector;
        TensorNet tensorNet;
        float* output;
        void* imgCPU;
        void* imgCUDA;
        float* output2python;
};


MobileNetSSD::MobileNetSSD()
{
    this->output2python = new float[600];
}

MobileNetSSD::~MobileNetSSD()
{
    cout << "mobileNetSSD's destructor called .." << endl;
}

void MobileNetSSD::init()
{
    const char* model  = "/home/ubuntu/newdisk/mobile-ssd/MobileNet-SSD-TensorRT/model/MobileNetSSD_deploy_iplugin.prototxt";
    const char* weight = "/home/ubuntu/newdisk/mobile-ssd/MobileNet-SSD-TensorRT/model/MobileNetSSD_deploy.caffemodel";
    
    this->INPUT_BLOB_NAME = "data";
    
    this->OUTPUT_BLOB_NAME = "detection_out";
    //this->BATCH_SIZE = 1;
    
    this->imageBuffer = new ConsumerProducerQueue<cv::Mat>(10,false);
    
    this->output_vector = {OUTPUT_BLOB_NAME};
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    float* data    = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    this->output  = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
}





/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* MobileNetSSD::allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}


void MobileNetSSD::loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}



cv::Mat MobileNetSSD::load_matrix_from_python(int * matrix, int rows, int cols, int channels){
    
 
    int i, j, c;
    cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(0,0,0));
    uchar* pxvec = img.ptr<uchar>(0);
    int count = 0;
    for(i = 0; i < rows; i++){
        pxvec = img.ptr<uchar>(i);
        for(j = 0; j < cols; j++){
            for(c = 0; c < channels; c++){
                pxvec[j*3+c] = (unsigned char)matrix[count];
                count = count + 2;
            }            
        }
    }
   return img;
}


//int main(int argc, char *argv[])
int MobileNetSSD::detect(int * matrix, int rows, int cols, int channels)
{
    int height = 300;
    int width  = 300;

    cv::Mat frame, srcImg;
    //frame = cv::imread(imgFile);
    frame = load_matrix_from_python(matrix, rows, cols, channels);

    srcImg = frame.clone();
    cv::resize(frame, frame, cv::Size(300,300));
    const size_t size = width * height * sizeof(float3);

    if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
    {
        cout <<"Cuda Memory allocation error occured."<<endl;
        return false;
    }

    void* imgData = malloc(size);
    memset(imgData,0,size);

    loadImg(frame,height,width,(float*)imgData,make_float3(127.5,127.5,127.5),0.007843);
    cudaMemcpyAsync(imgCUDA,imgData,size,cudaMemcpyHostToDevice);

    void* buffers[] = { imgCUDA, output };

    tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);

    vector<vector<float> > detections;

    int classIndex = 0;
    int flag = 0;
    for (int k=0; k<100; k++)
    {
        if(output[7*k+1] == -1){
            output2python[0] = k;
            break;
        }
//        float classIndex = output[7*k+1];
//        float confidence = output[7*k+2];
        float xmin = output[7*k + 3];
        float ymin = output[7*k + 4];
        float xmax = output[7*k + 5];
        float ymax = output[7*k + 6];
//        std::cout << classIndex << " , " << confidence << " , "  << xmin << " , " << ymin<< " , " << xmax<< " , " << ymax << std::endl;
        int x1 = static_cast<int>(xmin * srcImg.cols);
        int y1 = static_cast<int>(ymin * srcImg.rows);
        int x2 = static_cast<int>(xmax * srcImg.cols);
        int y2 = static_cast<int>(ymax * srcImg.rows);
        cv::rectangle(srcImg,cv::Rect2f(cv::Point(x1,y1),cv::Point(x2,y2)),cv::Scalar(255,0,255),1);
        output2python[6*k + 1] = output[7*k+1];
        output2python[6*k + 2] = output[7*k+2];
        output2python[6*k + 3] = x1;
        output2python[6*k + 4] = y1;
        output2python[6*k + 5] = x2;
        output2python[6*k + 6] = y2;
        
    }
    free(imgData);
}


void MobileNetSSD::destory(){
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
  
}

float* MobileNetSSD::getOutput(){
    return this->output2python;
}


extern "C"{

    MobileNetSSD* new_mobileNetSSD(){
        return new MobileNetSSD();
    } 

    void initMobileNetSSD(MobileNetSSD* mobileNetSSD){
        mobileNetSSD->init();
    }


//    void load_img(MobileNetSSD* mobileNetSSD, int * matrix, int rows, int cols, int channels){
//        mobileNetSSD->load_matrix_from_python(matrix, rows, cols, channels);
//    }

    void inferDetect(MobileNetSSD* mobileNetSSD, int * matrix, int rows, int cols, int channels){
        mobileNetSSD->detect(matrix, rows, cols, channels);
    }

    void destorySSD(MobileNetSSD* mobileNetSSD){
        mobileNetSSD->destory();
        delete mobileNetSSD;
    }

    void getOutput(MobileNetSSD* mobileNetSSD){
        mobileNetSSD->getOutput();
    }
}
