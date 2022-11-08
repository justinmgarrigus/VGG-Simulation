#ifndef IO_H 
#define IO_H 

#include <cassert>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <array>
#include <string>

extern "C" {
#include "ndarray.h" 
}

template <typename T> 
class RawDataset {
public:
    RawDataset() = default;
	
    RawDataset(uint32_t rows, uint32_t cols) {
		m_data = new T[rows * cols];

		m_dims[0] = rows;
		m_dims[1] = cols;
	}
	
	RawDataset(ndarray* const nd) {
		assert(nd->dim == 2); 
		
		m_data = new T[nd->count]; 
		m_dims[0] = (uint32_t)nd->shape[0]; 
		m_dims[1] = (uint32_t)nd->shape[1]; 
		memcpy(m_data, nd->arr, sizeof(T) * nd->count); 
	}
	
	RawDataset(const T* data, uint32_t rows, uint32_t cols)
		: RawDataset<T>::RawDataset(rows, cols) 
	{
		memcpy(m_data, data, sizeof(T) * rows * cols); 
	}
	
	RawDataset(std::string const &file) {
		load(file);
	}		

    RawDataset(RawDataset const &) = delete;

    ~RawDataset() {
        delete[] m_data;
    }

    void load(std::string const &file) {
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
    
	void save(std::string const &file) {
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

    uint32_t rows() const {
        return m_dims[0];
    }

    uint32_t cols() const {
        return m_dims[1];
    }
	
	T* data() const {
		return m_data; 
	}
	
	float entropy() {
		float sum = 0; 
		for (int i = 0; i < m_dims[0] * m_dims[1]; i++) 
			sum += (float)m_data[i]; 
		return sum; 
	}

    RawDataset &operator=(RawDataset const &) = delete;

    T &operator[](size_t index) {
        return m_data[index];
    }

    T const &operator[](size_t index) const {
        return m_data[index];
    }

private:
    T *m_data = nullptr;
    std::array<uint32_t, 2> m_dims;
};

#endif 