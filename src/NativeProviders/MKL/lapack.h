#pragma once

#include <memory>
#include "mkl_service.h"

const int INSUFFICIENT_MEMORY = -999999;
const int ALIGNMENT = 64;

struct mklfree
{
	void operator()(void* x) { mkl_free(x); }
};

template <typename T> using ptr = std::unique_ptr < T, mklfree > ;

template<typename T>
inline ptr<T> array_new(const int size, int alignment = ALIGNMENT) {
	auto ret = static_cast<T*>(mkl_malloc(size * sizeof(T), alignment));
	if (!ret)
	{
		throw new std::bad_alloc();
	}
	return ptr<T>(ret);
}

template<typename T>
inline ptr<T> array_clone(const int size, const T* array){
	auto clone = array_new<T>(size);
	memcpy(clone.get(), array, size * sizeof(T));
	return clone;
}