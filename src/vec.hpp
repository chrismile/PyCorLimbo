/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PYCORIANDER_VEC_HPP
#define PYCORIANDER_VEC_HPP

namespace math {

typedef int length_t;
template<length_t L, typename T> struct vec;

template<typename T>
struct vec<1, T>
{
    using vt = vec<1, T>;
    union {
        struct {
            T x;
        };
        T data[1];
    };

    vec() {}
    vec(float x) : x(x) {}
    T& operator[](int i) { return data[i]; }
    T operator[](int i) const { return data[i]; }
    vt operator+(vt rhs) const { return vt(x + rhs.x); }
    vt operator-(vt rhs) const { return vt(x - rhs.x); }
};

template<typename T>
struct vec<2, T>
{
    using vt = vec<2, T>;
    union {
        struct {
            T x, y;
        };
        T data[2];
    };

    vec() {}
    vec(float x, float y) : x(x), y(y) {}
    T& operator[](int i) { return data[i]; }
    T operator[](int i) const { return data[i]; }
    vt operator+(vt rhs) const { return vt(x + rhs.x, y + rhs.y); }
    vt operator-(vt rhs) const { return vt(x - rhs.x, y - rhs.y); }
};

}

#endif //PYCORIANDER_VEC_HPP
