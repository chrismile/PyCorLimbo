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

#ifndef PYCORIANDER_DIGAMMA_HPP
#define PYCORIANDER_DIGAMMA_HPP

#include <cmath>
#include <cstdint>

/**
 * Lanczos approximation of digamma function using weights by Viktor T. Toth.
 * - digamma = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x) (https://en.wikipedia.org/wiki/Digamma_function)
 * - Lanczos approximation: https://www.rskey.org/CMS/index.php/the-library/11
 * - Weights: https://www.rskey.org/CMS/index.php/the-library/11
 *
 * This function could be extended for values < 1 by:
 * - float z = 1 - iz;
 * - if (iz < 1) return digammaValue - M_PI * cosf(M_PI * iz) / sinf(M_PI * iz);
 */
#define G (5.15f)
#define P0 (2.50662827563479526904f)
#define P1 (225.525584619175212544f)
#define P2 (-268.295973841304927459f)
#define P3 (80.9030806934622512966f)
#define P4 (-5.00757863970517583837f)
#define P5 (0.0114684895434781459556f)
inline float digamma(uint32_t iz) {
    if (iz == 1u) {
        return -0.57721566490153287f;
    }
    float z = float(iz);
    float zh = z - 0.5f;
    float z1 = z + 1.0f;
    float z2 = z + 2.0f;
    float z3 = z + 3.0f;
    float z4 = z + 4.0f;
    float ZP = P0 + P1 / z + P2 / z1 + P3 / z2 + P4 / z3 + P5 / z4;
    float dZP = P1 / (z * z) + P2 / (z1 * z1) + P3 / (z2 * z2) + P4 / (z3 * z3) + P5 / (z4 * z4);
    float digammaValue = logf(zh + G) + zh / (zh + G) - dZP / ZP - 1.0f;
    return digammaValue;
}

#endif //PYCORIANDER_DIGAMMA_HPP
