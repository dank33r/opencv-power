/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include "precomp.hpp"
#include "imgwarp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace opt_VSX
{

class resizeNNInvokerVSX2 :
    public ParallelLoopBody
{
public:
    resizeNNInvokerVSX2(const Mat& _src, Mat &_dst, int *_x_ofs, int _pix_size4, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), pix_size4(_pix_size4),
        ify(_ify)
    {
    }

    virtual void operator() (const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int vsxWidth = width - (width & 0x7);
        for(y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step*y;
            uchar* Dstart = D;
            int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.data + sy*src.step;
            v_uint16x8 CV_DECL_ALIGNED(64) pixels = v_setall_u16(0);
            for(x = 0; x < vsxWidth; x += 8)
            {
                pixels = {*(ushort*)(S + x_ofs[x + 0]),
                          *(ushort*)(S + x_ofs[x + 1]),
                          *(ushort*)(S + x_ofs[x + 2]),
                          *(ushort*)(S + x_ofs[x + 3]),
                          *(ushort*)(S + x_ofs[x + 4]),
                          *(ushort*)(S + x_ofs[x + 5]),
                          *(ushort*)(S + x_ofs[x + 6]),
                          *(ushort*)(S + x_ofs[x + 7])};
                v_store((ushort*)D, pixels);
                D += 16;
            }
            for(; x < width; x++)
            {
                *(ushort*)(Dstart + x*2) = *(ushort*)(S + x_ofs[x]);
            }
        }
    }

private:
    const Mat src;
    Mat dst;
    int* x_ofs, pix_size4;
    double ify;

    resizeNNInvokerVSX2(const resizeNNInvokerVSX2&);
    resizeNNInvokerVSX2& operator=(const resizeNNInvokerVSX2&);
};

class resizeNNInvokerVSX :
    public ParallelLoopBody
{
public:
    resizeNNInvokerVSX(const Mat& _src, Mat &_dst, int *_x_ofs, int _pix_size4, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), pix_size4(_pix_size4),
        ify(_ify)
    {
    }
    virtual void operator() (const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int sseWidth = width - (width & 0x3);
        for(y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step*y;
            uchar* Dstart = D;
            int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.data + sy*src.step;
            v_int32x4 CV_DECL_ALIGNED(64) pixels = v_setall_s32(0);
            for(x = 0; x < sseWidth; x += 4)
            {
                pixels = { *(int*)(S + x_ofs[x + 0]),
                           *(int*)(S + x_ofs[x + 1]),
                           *(int*)(S + x_ofs[x + 2]),
                           *(int*)(S + x_ofs[x + 3])};
                
                v_store((int*)D, pixels);
                D += 16;
            }
            for(; x < width; x++)
            {
                *(int*)(Dstart + x*4) = *(int*)(S + x_ofs[x]);
            }
        }
    }

private:
    const Mat src;
    Mat dst;
    int* x_ofs, pix_size4;
    double ify;

    resizeNNInvokerVSX(const resizeNNInvokerVSX&);
    resizeNNInvokerVSX& operator=(const resizeNNInvokerVSX&);
};

void resizeNN2_VSX(const Range& range, const Mat& src, Mat &dst, int *x_ofs, int pix_size4, double ify)
{
    resizeNNInvokerVSX2 invoker(src, dst, x_ofs, pix_size4, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

void resizeNN4_VSX(const Range& range, const Mat& src, Mat &dst, int *x_ofs, int pix_size4, double ify)
{
    resizeNNInvokerVSX invoker(src, dst, x_ofs, pix_size4, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

int VResizeLanczos4Vec_32f16u_VSX(const uchar** _src, uchar* _dst, const uchar* _beta, int width)
{
    const float** src = (const float**)_src;
    const float* beta = (const float*)_beta;
    const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
        *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
    short * dst = (short*)_dst;
    int x = 0;
    v_float32x4 v_b0 = v_setall_f32(beta[0]), v_b1 = v_setall_f32(beta[1]),
        v_b2 = v_setall_f32(beta[2]), v_b3 = v_setall_f32(beta[3]),
        v_b4 = v_setall_f32(beta[4]), v_b5 = v_setall_f32(beta[5]),
        v_b6 = v_setall_f32(beta[6]), v_b7 = v_setall_f32(beta[7]);

    for (; x <= width - 8; x += 8)
    {
        v_float32x4 v_dst0 = (v_b0 * v_load(S0 + x));
        v_dst0 = v_dst0 + v_b1 * v_load(S1 + x);
        v_dst0 = v_dst0 + v_b2 * v_load(S2 + x);
        v_dst0 = v_dst0 + v_b3 * v_load(S3 + x);
        v_dst0 = v_dst0 + v_b4 * v_load(S4 + x);
        v_dst0 = v_dst0 + v_b5 * v_load(S5 + x);
        v_dst0 = v_dst0 + v_b6 * v_load(S6 + x);
        v_dst0 = v_dst0 + v_b7 * v_load(S7 + x);

        v_float32x4 v_dst1 = v_b0 * v_load(S0 + x + 4);
        v_dst1 = v_dst1 + v_b1 * v_load(S1 + x + 4);
        v_dst1 = v_dst1 + v_b2 * v_load(S2 + x + 4);
        v_dst1 = v_dst1 + v_b3 * v_load(S3 + x + 4);
        v_dst1 = v_dst1 + v_b4 * v_load(S4 + x + 4);
        v_dst1 = v_dst1 + v_b5 * v_load(S5 + x + 4);
        v_dst1 = v_dst1 + v_b6 * v_load(S6 + x + 4);
        v_dst1 = v_dst1 + v_b7 * v_load(S7 + x + 4);

        v_int32x4 v_dsti0 = v_round(v_dst0);
        v_int32x4 v_dsti1 = v_round(v_dst1);

        v_store((dst + x), v_pack(v_dsti0, v_dsti1));
    }

    return x;
}

void convertMaps_nninterpolate32f1c16s_VSX(const float* src1f, const float* src2f, short* dst1, int width)
{
    int x = 0;
    for (; x <= width - 16; x += 16)
    {
        v_int16x8 v_dst0 = v_pack(v_round(v_load(src1f + x)),
            v_round(v_load(src1f + x + 4)));
        v_int16x8 v_dst1 = v_pack(v_round(v_load(src1f + x + 8)),
            v_round(v_load(src1f + x + 12)));

        v_int16x8 v_dst2 = v_pack(v_round(v_load(src2f + x)),
            v_round(v_load(src2f + x + 4)));
        v_int16x8 v_dst3 = v_pack(v_round(v_load(src2f + x + 8)),
            v_round(v_load(src2f + x + 12)));

        v_store_interleave(dst1 + x * 2, v_dst0, v_dst1, v_dst2, v_dst3);

    }

    for (; x < width; x++)
    {
        dst1[x * 2] = saturate_cast<short>(src1f[x]);
        dst1[x * 2 + 1] = saturate_cast<short>(src2f[x]);
    }
}

void convertMaps_32f1c16s_VSX(const float* src1f, const float* src2f, short* dst1, ushort* dst2, int width)
{
    int x = 0;
    v_float32x4 v_its = v_setall_f32(INTER_TAB_SIZE);
    v_int32x4   v_its1 = v_setall_s32(INTER_TAB_SIZE - 1);

    for (; x <= width - 16; x += 16)
    {
        v_int32x4 v_ix0 = v_round(v_load(src1f + x)     * v_its);
        v_int32x4 v_ix1 = v_round(v_load(src1f + x + 4) * v_its);
        v_int32x4 v_iy0 = v_round(v_load(src2f + x)     * v_its);
        v_int32x4 v_iy1 = v_round(v_load(src2f + x + 4) * v_its);

        v_int16x8 v_dst10 = v_pack(v_ix0 >> INTER_BITS,v_ix1 >> INTER_BITS);
        v_int16x8 v_dst12 = v_pack(v_iy0 >> INTER_BITS,v_iy1 >> INTER_BITS);
        v_int32x4 v_dst20 = (v_iy0 & v_its1) * v_setall_s32(INTER_TAB_SIZE) + (v_ix0 & v_its1);
        v_int32x4 v_dst21 = (v_iy1 & v_its1) * v_setall_s32(INTER_TAB_SIZE) + (v_ix1 & v_its1);
        v_store_interleave((dst1 + x * 2), v_dst10, v_dst12);
        v_store((dst2 + x), v_pack_u(v_dst20, v_dst21));

        v_ix0 = v_round(v_load(src1f + x + 8) * v_its);
        v_ix1 = v_round(v_load(src1f + x + 12)* v_its);
        v_iy0 = v_round(v_load(src2f + x + 8) * v_its);
        v_iy1 = v_round(v_load(src2f + x + 12)* v_its);

        v_int16x8 v_dst11 = v_pack((v_ix0 >> INTER_BITS),(v_ix1 >> INTER_BITS));
        v_int16x8 v_dst13 = v_pack((v_iy0 >> INTER_BITS),(v_iy1 >> INTER_BITS));
        v_dst20 = ((v_iy0 & v_its1) * v_setall_s32(INTER_TAB_SIZE)) + (v_ix0 & v_its1);
        v_dst21 = ((v_iy1 & v_its1) * v_setall_s32(INTER_TAB_SIZE)) + (v_ix1 & v_its1);
        v_store_interleave((dst1 + x * 2 + 16), v_dst11, v_dst13);
        v_store(dst2 + x + 8, v_pack_u(v_dst20, v_dst21));

    }
    for (; x < width; x++)
    {
        int ix = saturate_cast<int>(src1f[x] * INTER_TAB_SIZE);
        int iy = saturate_cast<int>(src2f[x] * INTER_TAB_SIZE);
        dst1[x * 2] = saturate_cast<short>(ix >> INTER_BITS);
        dst1[x * 2 + 1] = saturate_cast<short>(iy >> INTER_BITS);
        dst2[x] = (ushort)((iy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE - 1)));
    }
}
void convertMaps_32f2c16s_VSX(const float* src1f, short* dst1, ushort* dst2, int width)
{
    int x = 0;
    v_float32x4 v_its = v_setall_f32(INTER_TAB_SIZE);
    v_int32x4 v_its1 = v_setall_s32(INTER_TAB_SIZE - 1);

    for (; x <= width - 8; x += 8)
    {

        v_float32x4 v_fx0, v_fx1, v_fy0, v_fy1;
        v_load_deinterleave(src1f + x * 2, v_fx0, v_fy0);
        v_load_deinterleave(src1f + x * 2 + 8, v_fx1, v_fy1);
        v_int32x4 v_x0 = v_round(v_fx0 * v_its);
        v_int32x4 v_y0 = v_round(v_fy0 * v_its);
        v_int32x4 v_x1 = v_round(v_fx1 * v_its);
        v_int32x4 v_y1 = v_round(v_fy1 * v_its);

        v_int16x8 v_dx = v_pack((v_x0 >> INTER_BITS),(v_x1 >> INTER_BITS));
        v_int16x8 v_dy = v_pack((v_y0 >> INTER_BITS),(v_y1 >> INTER_BITS));
        v_store_interleave((dst1 + x * 2), v_dx, v_dy);

        v_x0 = (v_x0 & v_its1) + (v_y0 & v_its1) * v_setall_s32(INTER_TAB_SIZE);
        v_x1 = (v_x1 & v_its1) + (v_y1 & v_its1) * v_setall_s32(INTER_TAB_SIZE);
        v_store(dst2 + x, v_pack_u(v_x0, v_x1));
    }

    for (; x < width; x++)
    {
        int ix = saturate_cast<int>(src1f[x * 2] * INTER_TAB_SIZE);
        int iy = saturate_cast<int>(src1f[x * 2 + 1] * INTER_TAB_SIZE);
        dst1[x * 2] = saturate_cast<short>(ix >> INTER_BITS);
        dst1[x * 2 + 1] = saturate_cast<short>(iy >> INTER_BITS);
        dst2[x] = (ushort)((iy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE - 1)));
    }
}

void WarpAffineInvoker_Blockline_VSX(int *adelta, int *bdelta, short* xy, int X0, int Y0, int bw)
{
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    int x1 = 0;

    v_int32x4 v_X0 = v_setall_s32(X0);
    v_int32x4 v_Y0 = v_setall_s32(Y0);
    for (; x1 < bw - 16; x1 += 16)
    {
        v_int16x8 v_x0 = v_pack((v_X0 + v_load(adelta + x1)) >> AB_BITS,
                                (v_X0 + v_load(adelta + x1 + 4)) >>  AB_BITS);
        v_int16x8 v_x1 = v_pack((v_X0 + v_load(adelta + x1 + 8)) >> AB_BITS,
                                (v_X0 + v_load(adelta + x1 + 12)) >> AB_BITS);

        v_int16x8 v_y0 = v_pack((v_Y0 + v_load(bdelta + x1)) >> AB_BITS,
                                (v_Y0 + v_load(bdelta + x1 + 4)) >> AB_BITS);
        v_int16x8 v_y1 = v_pack((v_Y0 + v_load(bdelta + x1 + 8)) >> AB_BITS,
                                (v_Y0 + v_load(bdelta + x1 + 12))>> AB_BITS);

        v_store_interleave((xy + x1 * 2),v_x0, v_y0);
        v_store_interleave( xy + x1*2 + 16, v_x1, v_y1);
    }
    for (; x1 < bw; x1++)
    {
        int X = (X0 + adelta[x1]) >> AB_BITS;
        int Y = (Y0 + bdelta[x1]) >> AB_BITS;
        xy[x1 * 2] = saturate_cast<short>(X);
        xy[x1 * 2 + 1] = saturate_cast<short>(Y);
    }

}


class WarpPerspectiveLine_VSX_Impl: public WarpPerspectiveLine_VSX
{
public:
    WarpPerspectiveLine_VSX_Impl(const double *M)
    {
        CV_UNUSED(M);
    }
    virtual void processNN(const double *M, short* xy, double X0, double Y0, double W0, int bw)
    {
        const v_float64x2 v_M0 = v_setall_f64(M[0]);
        const v_float64x2 v_M3 = v_setall_f64(M[3]);
        const v_float64x2 v_M6 = v_setall_f64(M[6]);
        const v_float64x2 v_intmax = v_setall_f64((double)INT_MAX);
        const v_float64x2 v_intmin = v_setall_f64((double)INT_MIN);
        const v_float64x2 v_2 = v_setall_f64(2);
        const v_float64x2 v_zero = v_setall_f64(0);
        const v_float64x2 v_1 = v_setall_f64(1);

        int x1 = 0;
        v_float64x2 v_X0d = v_setall_f64(X0);
        v_float64x2 v_Y0d = v_setall_f64(Y0);
        v_float64x2 v_W0 = v_setall_f64(W0);
        v_float64x2 v_x1;
        v_x1 = {0.0, 1.0};

        for (; x1 <= bw - 16; x1 += 16)
        {
            // 0-3
            v_int32x4 v_X0, v_Y0;
            {
                v_float64x2 v_W = v_M6 * v_x1 + v_W0;
                v_W = (~(v_W == v_zero)) & (v_1 / v_W);
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1)* v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1)* v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 + v_W0;
                v_W = (~(v_W == v_zero)) & (v_1 / v_W);
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X0 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y0 = v_combine_low(v_round(v_fY0), v_round(v_fY1));
            }

            // 4-8
            v_int32x4 v_X1, v_Y1;
            {
                v_float64x2 v_W = v_M6 * v_x1 + v_W0;
                v_W = ~(v_W == v_zero) & (v_1 / v_W);
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 + v_W0;
                v_W = ~(v_W == v_zero) & (v_1 / v_W);
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X1 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y1 = v_combine_low(v_round(v_fY0), v_round(v_fY1));
            }

            // 8-11
            v_int32x4 v_X2, v_Y2;
            {
                v_float64x2 v_W = v_M6 * v_x1 + v_W0;
                v_W = ~(v_W == v_zero) & (v_1 / v_W);
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 + v_W0;
                v_W = ~(v_W == v_zero) & (v_1 / v_W);
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X2 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y2 = v_combine_low(v_round(v_fY0), v_round(v_fY1));
            }

            // 12-15
            v_int32x4 v_X3, v_Y3;
            {
                v_float64x2 v_W = v_M6 * v_x1 + v_W0;
                v_W = ~(v_W == v_zero) & (v_1 / v_W);
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 + v_W0;
                v_W = ~(v_W == v_zero) & (v_1 / v_W);
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X3 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y3 = v_combine_low(v_round(v_fY0), v_round(v_fY1));
            }

            // convert to 16s
            v_int16x8 r_X0 = v_pack(v_X0, v_X1);
            v_int16x8 r_X1 = v_pack(v_X2, v_X3);
            v_int16x8 r_Y0 = v_pack(v_Y0, v_Y1);
            v_int16x8 r_Y1 = v_pack(v_Y2, v_Y3);

            v_store_interleave(xy + x1 * 2, r_X0, r_Y0);
            v_store_interleave(xy + x1*2+16, r_X1, r_Y1);
        }

        for (; x1 < bw; x1++)
        {
            double W = W0 + M[6] * x1;
            W = W ? 1. / W : 0;
            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1)*W));
            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1)*W));
            int X = saturate_cast<int>(fX);
            int Y = saturate_cast<int>(fY);

            xy[x1 * 2] = saturate_cast<short>(X);
            xy[x1 * 2 + 1] = saturate_cast<short>(Y);
        }
    }
    virtual void process(const double *M, short* xy, short* alpha, double X0, double Y0, double W0, int bw)
    {
        const v_float64x2 v_M0 = v_setall_f64(M[0]);
        const v_float64x2 v_M3 = v_setall_f64(M[3]);
        const v_float64x2 v_M6 = v_setall_f64(M[6]);
        const v_float64x2 v_intmax = v_setall_f64((double)INT_MAX);
        const v_float64x2 v_intmin = v_setall_f64((double)INT_MIN);
        const v_float64x2 v_2 = v_setall_f64(2.0);
        const v_float64x2 v_zero = v_setall_f64(0.0);
        const v_float64x2 v_its = v_setall_f64(INTER_TAB_SIZE);
        const v_int32x4 v_itsi1 = v_setall_s32(INTER_TAB_SIZE - 1);

        int x1 = 0;

        v_float64x2 v_X0d = v_setall_f64(X0);
        v_float64x2 v_Y0d = v_setall_f64(Y0);
        v_float64x2 v_W0 = v_setall_f64(W0);
        v_float64x2 v_x1;
        v_x1 = {0.0, 1.0};

        for (; x1 <= bw - 32; x1 += 16)
        {
            // 0-3
            v_int32x4 v_X0, v_Y0;
            {
                v_float64x2 v_W = (v_M6 * v_x1) + v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 +  v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X0 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y0 = v_combine_low(v_round(v_fY0), v_round(v_fY1));
            }

            // 4-8
            v_int32x4 v_X1, v_Y1;
            {
                v_float64x2 v_W = (v_M6 * v_x1) + v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 +  v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X1 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y1 = v_combine_low(v_round(v_fY0), v_round(v_fY1));
            }
  
            // 8-11
            v_int32x4 v_X2, v_Y2;
            {
                v_float64x2 v_W = (v_M6 * v_x1) + v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 +  v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X2 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y2 = v_combine_low(v_round(v_fY0), v_round(v_fY1));

            }

            // 12-15
            v_int32x4 v_X3, v_Y3;
            {
                v_float64x2 v_W = (v_M6 * v_x1) + v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_W = v_M6 * v_x1 +  v_W0;
                v_W = v_select((v_W == v_zero), v_zero,  (v_its / v_W));
                v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, (v_X0d + v_M0 * v_x1) * v_W));
                v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, (v_Y0d + v_M3 * v_x1) * v_W));
                v_x1 = v_x1 + v_2;

                v_X3 = v_combine_low(v_round(v_fX0), v_round(v_fX1));
                v_Y3 = v_combine_low(v_round(v_fY0), v_round(v_fY1));
            }

            // store alpha
            v_int32x4 alpha0_tmp = (v_Y0 & v_itsi1) * v_setall_s32(INTER_TAB_SIZE) + (v_X0 & v_itsi1);
            v_int32x4 alpha1_tmp = (v_Y1 & v_itsi1) * v_setall_s32(INTER_TAB_SIZE) + (v_X1 & v_itsi1);
            v_store(alpha + x1, v_pack(alpha0_tmp, alpha1_tmp));

            alpha0_tmp = (v_Y2 & v_itsi1) * v_setall_s32(INTER_TAB_SIZE) + (v_X2 & v_itsi1);
            alpha1_tmp = (v_Y3 & v_itsi1) * v_setall_s32(INTER_TAB_SIZE) + (v_X3 & v_itsi1);
            v_store(alpha + x1 + 8, v_pack(alpha0_tmp, alpha1_tmp));

            // convert to 16s
            v_int16x8 tmp_x0 = v_pack((v_X0 >> INTER_BITS),(v_X1 >> INTER_BITS));
            v_int16x8 tmp_x1 = v_pack((v_X2 >> INTER_BITS),(v_X3 >> INTER_BITS));
            v_int16x8 tmp_y0 = v_pack((v_Y0 >> INTER_BITS),(v_Y1 >> INTER_BITS));
            v_int16x8 tmp_y1 = v_pack((v_Y2 >> INTER_BITS),(v_Y3 >> INTER_BITS));

            v_store_interleave(xy + x1 * 2, tmp_x0, tmp_y0);
            v_store_interleave(xy + x1*2+16, tmp_x1, tmp_y1);
        }

        for (; x1 < bw; x1++)
        {
            double W = W0 + M[6] * x1;
            W = W ? INTER_TAB_SIZE / W : 0;
            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1)*W));
            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1)*W));
            int X = saturate_cast<int>(fX);
            int Y = saturate_cast<int>(fY);

            xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);
            xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);
            alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE +
                (X & (INTER_TAB_SIZE - 1)));
        }
    }
    virtual ~WarpPerspectiveLine_VSX_Impl() {};
};

Ptr<WarpPerspectiveLine_VSX> WarpPerspectiveLine_VSX::getImpl(const double *M)
{
    return Ptr<WarpPerspectiveLine_VSX>(new WarpPerspectiveLine_VSX_Impl(M));
}

}
}
/* End of file. */
