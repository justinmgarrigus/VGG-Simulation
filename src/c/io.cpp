#include "io.h" 

#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include <tools/util/host_matrix.h>

#include <cassert>
#include <fstream>
#include <getopt.h>
#include <iostream>

template <typename T> 
RawDataset<T>::RawDataset(uint32_t rows, uint32_t cols) {
	m_data = new T[rows * cols];

	m_dims[0] = rows;
	m_dims[1] = cols;
}

template <typename T> 
RawDataset<T>::RawDataset(ndarray* const nd) {
	assert(nd->dim == 2); 
	
	m_data = new T[nd->count]; 
	m_dims[0] = nd->shape[0]; 
	m_dims[1] = nd->shape[1]; 
	memcpy(m_data, nd->arr, sizeof(T) * nd->count); 
}

template <typename T>
RawDataset<T>::RawDataset(const T* data, uint32_t rows, uint32_t cols) 
	: RawDataset<T>::RawDataset(rows, cols) 
{
	memcpy(m_data, data, sizeof(T) * rows * cols); 
}

template <typename T> 
void RawDataset<T>::load(std::string const &file) {
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs) {
        throw std::system_error(
            errno,
            std::system_category(),
            "Failed to open \"" + file + "\""
        );
    }

    ifs.read(reinterpret_cast<char *>(m_dims.data()), sizeof(uint32_t) * 2);

    auto fsize = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    fsize = ifs.tellg() - fsize;

    ifs.clear();
    ifs.seekg(sizeof(uint32_t) * 4, std::ios::beg);

    auto count = fsize / sizeof(T);

    m_data = new T[count];

    for (size_t i = 0; i < count; ++i) {
        T tmp;
        ifs.read(reinterpret_cast<char *>(&tmp), sizeof(T));
        m_data[i] = tmp;
    }
}

template <typename T> 
void RawDataset<T>::save(std::string const &file) {
	std::cout << "Saving file: " << file << std::endl; 
    std::ofstream ofs(file, std::ios::binary);
    if (!ofs) {
        throw std::system_error(
            errno,
            std::system_category(),
            "Failed to open for write \"" + file + "\""
        );
    }

    ofs.write(reinterpret_cast<char const *>(m_dims.data()), sizeof(uint32_t) * m_dims.size());
    ofs.write(reinterpret_cast<char const *>(m_data), sizeof(T) * rows() * cols());

    ofs.close();
}

using WmmaGemmTraits = cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<64, 32, 32>,
    half,
    half,
    float,
    cutlass::gemm::LinearScaling<float>,
    float,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<
        typename cutlass::Shape<64, 32, 16>
    >::Shape,
    cutlass::Shape<16, 16, 16>
>;

using Gemm = cutlass::gemm::Gemm<WmmaGemmTraits>;
typename Gemm::Params params;

//int main(int argc, char **argv) {
//    option const long_opts[] = {
//        { "input", required_argument, nullptr, 'x' },
//        { "weight", required_argument, nullptr, 'w' },
//        { nullptr, no_argument, nullptr, 0 },
//    };
//
//    RawDataset x;
//    RawDataset w;
//
//    int opt;
//    while ((opt = getopt_long(argc, argv, "x:w:y:", long_opts, nullptr)) != -1) {
//        switch (opt) {
//        case 'x':
//            x.load(optarg);
//            break;
//        case 'w':
//            w.load(optarg);
//            break;
//        }
//    }
//
//    // Dimensions must be compatible for multiplication
//    assert(w.cols() == x.rows());
//
//    cutlass::HostMatrixRowMajor<cutlass::half_t> A(cutlass::MatrixCoord(w.rows(), w.cols()));
//    for (size_t i = 0; i < w.rows() * w.cols(); ++i) {
//        A.host_data()[i] = cutlass::half_t::convert(w[i]);
//    }
//    A.sync_device();
//
//    cutlass::HostMatrixRowMajor<cutlass::half_t> B(cutlass::MatrixCoord(x.rows(), x.cols()));
//    for (size_t i = 0; i < x.rows() * x.cols(); ++i) {
//        B.host_data()[i] = cutlass::half_t::convert(x[i]);
//    }
//    B.sync_device();
//
//    cutlass::HostMatrix<float> C(cutlass::MatrixCoord(w.rows(), x.cols()));
//
//    params.initialize(
//        w.rows(),
//        x.cols(),
//        w.cols(),
//        1.0f,
//        A.device_data(),
//        A.leading_dim(),
//        B.device_data(),
//        B.leading_dim(),
//        0.0f,
//        C.device_data(),
//        C.leading_dim(),
//        C.device_data(),
//        C.leading_dim()
//    );
//
//    Gemm::launch(params);
//
//    C.sync_host();
//}