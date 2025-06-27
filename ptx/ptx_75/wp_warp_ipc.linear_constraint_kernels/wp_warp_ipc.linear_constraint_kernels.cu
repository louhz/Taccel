
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

extern "C" {
}


extern "C" __global__ void linear_constraint_project_system_kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::int32> var_columns,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_values,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient,
    wp::int32 var_n_reduced_dofs)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3,wp::float64>* var_1;
        wp::vec_t<3,wp::float64> var_2;
        wp::vec_t<3,wp::float64> var_3;
        const wp::int32 var_4 = 0;
        const wp::int32 var_5 = 3;
        wp::int32 var_6;
        wp::int32 var_7;
        bool var_8;
        const wp::float64 var_9 = 0.0;
        const wp::int32 var_10 = 1;
        const wp::int32 var_11 = 3;
        wp::int32 var_12;
        wp::int32 var_13;
        bool var_14;
        const wp::int32 var_15 = 2;
        const wp::int32 var_16 = 3;
        wp::int32 var_17;
        wp::int32 var_18;
        bool var_19;
        wp::int32* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::range_t var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::mat_t<3,3,wp::float64>* var_31;
        wp::mat_t<3,3,wp::float64> var_32;
        wp::mat_t<3,3,wp::float64> var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 3;
        wp::int32 var_36;
        wp::int32 var_37;
        bool var_38;
        const wp::int32 var_39 = 0;
        const wp::int32 var_40 = 1;
        const wp::int32 var_41 = 2;
        const wp::int32 var_42 = 3;
        wp::int32 var_43;
        wp::int32 var_44;
        bool var_45;
        const wp::int32 var_46 = 0;
        const wp::int32 var_47 = 1;
        const wp::int32 var_48 = 2;
        bool var_49;
        bool var_50;
        bool var_51;
        const wp::float64 var_52 = 1.0;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 3;
        wp::int32 var_55;
        wp::int32 var_56;
        bool var_57;
        const wp::int32 var_58 = 0;
        const wp::int32 var_59 = 1;
        const wp::int32 var_60 = 2;
        wp::int32 var_61;
        const wp::int32 var_62 = 3;
        wp::int32 var_63;
        wp::int32 var_64;
        bool var_65;
        const wp::int32 var_66 = 0;
        const wp::int32 var_67 = 1;
        const wp::int32 var_68 = 2;
        wp::int32 var_69;
        bool var_70;
        bool var_71;
        bool var_72;
        const wp::int32 var_73 = 2;
        const wp::int32 var_74 = 3;
        wp::int32 var_75;
        wp::int32 var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        const wp::int32 var_79 = 1;
        const wp::int32 var_80 = 2;
        wp::int32 var_81;
        const wp::int32 var_82 = 3;
        wp::int32 var_83;
        wp::int32 var_84;
        bool var_85;
        const wp::int32 var_86 = 0;
        const wp::int32 var_87 = 1;
        const wp::int32 var_88 = 2;
        wp::int32 var_89;
        bool var_90;
        bool var_91;
        bool var_92;
        //---------
        // forward
        // def linear_constraint_project_system_kernel(                                           <L 7>
        // block_row_idx = wp.tid()                                                               <L 17>
        var_0 = builtin_tid1d();
        // grad_seg = gradient[block_row_idx]                                                     <L 18>
        var_1 = wp::address(var_gradient, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // for d_row in range(3):                                                                 <L 19>
        // row_idx = block_row_idx * 3 + d_row                                                    <L 20>
        var_6 = wp::mul(var_0, var_5);
        var_7 = wp::add(var_6, var_4);
        // if row_idx < n_reduced_dofs:                                                           <L 21>
        var_8 = (var_7 < var_n_reduced_dofs);
        if (var_8) {
            // grad_seg[d_row] = _0                                                               <L 22>
            wp::assign_inplace(var_2, var_4, var_9);
        }
        // row_idx = block_row_idx * 3 + d_row                                                    <L 20>
        var_12 = wp::mul(var_0, var_11);
        var_13 = wp::add(var_12, var_10);
        // if row_idx < n_reduced_dofs:                                                           <L 21>
        var_14 = (var_13 < var_n_reduced_dofs);
        if (var_14) {
            // grad_seg[d_row] = _0                                                               <L 22>
            wp::assign_inplace(var_2, var_10, var_9);
        }
        // row_idx = block_row_idx * 3 + d_row                                                    <L 20>
        var_17 = wp::mul(var_0, var_16);
        var_18 = wp::add(var_17, var_15);
        // if row_idx < n_reduced_dofs:                                                           <L 21>
        var_19 = (var_18 < var_n_reduced_dofs);
        if (var_19) {
            // grad_seg[d_row] = _0                                                               <L 22>
            wp::assign_inplace(var_2, var_15, var_9);
        }
        // gradient[block_row_idx] = grad_seg                                                     <L 23>
        wp::array_store(var_gradient, var_0, var_2);
        // for k in range(offsets[block_row_idx], offsets[block_row_idx + 1]):                    <L 25>
        var_20 = wp::address(var_offsets, var_0);
        var_22 = wp::add(var_0, var_21);
        var_23 = wp::address(var_offsets, var_22);
        var_25 = wp::load(var_20);
        var_26 = wp::load(var_23);
        var_24 = wp::range(var_25, var_26);
        start_for_0:;
            if (iter_cmp(var_24) == 0) goto end_for_0;
            var_27 = wp::iter_next(var_24);
            // block_col_idx = columns[k]                                                         <L 26>
            var_28 = wp::address(var_columns, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // block_val = values[k]                                                              <L 27>
            var_31 = wp::address(var_values, var_27);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // for d in range(3):                                                                 <L 28>
            // row_idx = block_row_idx * 3 + d                                                    <L 29>
            var_36 = wp::mul(var_0, var_35);
            var_37 = wp::add(var_36, var_34);
            // if row_idx < n_reduced_dofs:                                                       <L 30>
            var_38 = (var_37 < var_n_reduced_dofs);
            if (var_38) {
                // for dj in range(3):                                                            <L 31>
                // block_val[d, dj] = _0                                                          <L 32>
                wp::assign_inplace(var_32, var_34, var_39, var_9);
                wp::assign_inplace(var_32, var_34, var_40, var_9);
                wp::assign_inplace(var_32, var_34, var_41, var_9);
            }
            // col_idx = block_col_idx * 3 + d                                                    <L 34>
            var_43 = wp::mul(var_29, var_42);
            var_44 = wp::add(var_43, var_34);
            // if col_idx < n_reduced_dofs:                                                       <L 35>
            var_45 = (var_44 < var_n_reduced_dofs);
            if (var_45) {
                // for di in range(3):                                                            <L 36>
                // block_val[di, d] = _0                                                          <L 37>
                wp::assign_inplace(var_32, var_46, var_34, var_9);
                wp::assign_inplace(var_32, var_47, var_34, var_9);
                wp::assign_inplace(var_32, var_48, var_34, var_9);
            }
            // if row_idx < n_reduced_dofs and row_idx == col_idx:                                <L 39>
            var_49 = (var_37 < var_n_reduced_dofs);
            var_50 = (var_37 == var_44);
            var_51 = var_49 && var_50;
            if (var_51) {
                // block_val[d, d] = _1                                                           <L 40>
                wp::assign_inplace(var_32, var_34, var_34, var_52);
            }
            // row_idx = block_row_idx * 3 + d                                                    <L 29>
            var_55 = wp::mul(var_0, var_54);
            var_56 = wp::add(var_55, var_53);
            // if row_idx < n_reduced_dofs:                                                       <L 30>
            var_57 = (var_56 < var_n_reduced_dofs);
            if (var_57) {
                // for dj in range(3):                                                            <L 31>
                // block_val[d, dj] = _0                                                          <L 32>
                wp::assign_inplace(var_32, var_53, var_58, var_9);
                wp::assign_inplace(var_32, var_53, var_59, var_9);
                wp::assign_inplace(var_32, var_53, var_60, var_9);
            }
            var_61 = wp::where(var_57, var_60, var_41);
            // col_idx = block_col_idx * 3 + d                                                    <L 34>
            var_63 = wp::mul(var_29, var_62);
            var_64 = wp::add(var_63, var_53);
            // if col_idx < n_reduced_dofs:                                                       <L 35>
            var_65 = (var_64 < var_n_reduced_dofs);
            if (var_65) {
                // for di in range(3):                                                            <L 36>
                // block_val[di, d] = _0                                                          <L 37>
                wp::assign_inplace(var_32, var_66, var_53, var_9);
                wp::assign_inplace(var_32, var_67, var_53, var_9);
                wp::assign_inplace(var_32, var_68, var_53, var_9);
            }
            var_69 = wp::where(var_65, var_68, var_48);
            // if row_idx < n_reduced_dofs and row_idx == col_idx:                                <L 39>
            var_70 = (var_56 < var_n_reduced_dofs);
            var_71 = (var_56 == var_64);
            var_72 = var_70 && var_71;
            if (var_72) {
                // block_val[d, d] = _1                                                           <L 40>
                wp::assign_inplace(var_32, var_53, var_53, var_52);
            }
            // row_idx = block_row_idx * 3 + d                                                    <L 29>
            var_75 = wp::mul(var_0, var_74);
            var_76 = wp::add(var_75, var_73);
            // if row_idx < n_reduced_dofs:                                                       <L 30>
            var_77 = (var_76 < var_n_reduced_dofs);
            if (var_77) {
                // for dj in range(3):                                                            <L 31>
                // block_val[d, dj] = _0                                                          <L 32>
                wp::assign_inplace(var_32, var_73, var_78, var_9);
                wp::assign_inplace(var_32, var_73, var_79, var_9);
                wp::assign_inplace(var_32, var_73, var_80, var_9);
            }
            var_81 = wp::where(var_77, var_80, var_61);
            // col_idx = block_col_idx * 3 + d                                                    <L 34>
            var_83 = wp::mul(var_29, var_82);
            var_84 = wp::add(var_83, var_73);
            // if col_idx < n_reduced_dofs:                                                       <L 35>
            var_85 = (var_84 < var_n_reduced_dofs);
            if (var_85) {
                // for di in range(3):                                                            <L 36>
                // block_val[di, d] = _0                                                          <L 37>
                wp::assign_inplace(var_32, var_86, var_73, var_9);
                wp::assign_inplace(var_32, var_87, var_73, var_9);
                wp::assign_inplace(var_32, var_88, var_73, var_9);
            }
            var_89 = wp::where(var_85, var_88, var_69);
            // if row_idx < n_reduced_dofs and row_idx == col_idx:                                <L 39>
            var_90 = (var_76 < var_n_reduced_dofs);
            var_91 = (var_76 == var_84);
            var_92 = var_90 && var_91;
            if (var_92) {
                // block_val[d, d] = _1                                                           <L 40>
                wp::assign_inplace(var_32, var_73, var_73, var_52);
            }
            // values[k] = block_val                                                              <L 41>
            wp::array_store(var_values, var_27, var_32);
            wp::assign(var_18, var_76);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void linear_constraint_project_system_kernel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::int32> var_columns,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_values,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient,
    wp::int32 var_n_reduced_dofs,
    wp::array_t<wp::int32> adj_offsets,
    wp::array_t<wp::int32> adj_columns,
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_values,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_gradient,
    wp::int32 adj_n_reduced_dofs)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3,wp::float64>* var_1;
        wp::vec_t<3,wp::float64> var_2;
        wp::vec_t<3,wp::float64> var_3;
        const wp::int32 var_4 = 0;
        const wp::int32 var_5 = 3;
        wp::int32 var_6;
        wp::int32 var_7;
        bool var_8;
        const wp::float64 var_9 = 0.0;
        const wp::int32 var_10 = 1;
        const wp::int32 var_11 = 3;
        wp::int32 var_12;
        wp::int32 var_13;
        bool var_14;
        const wp::int32 var_15 = 2;
        const wp::int32 var_16 = 3;
        wp::int32 var_17;
        wp::int32 var_18;
        bool var_19;
        wp::int32* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::range_t var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::mat_t<3,3,wp::float64>* var_31;
        wp::mat_t<3,3,wp::float64> var_32;
        wp::mat_t<3,3,wp::float64> var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 3;
        wp::int32 var_36;
        wp::int32 var_37;
        bool var_38;
        const wp::int32 var_39 = 0;
        const wp::int32 var_40 = 1;
        const wp::int32 var_41 = 2;
        const wp::int32 var_42 = 3;
        wp::int32 var_43;
        wp::int32 var_44;
        bool var_45;
        const wp::int32 var_46 = 0;
        const wp::int32 var_47 = 1;
        const wp::int32 var_48 = 2;
        bool var_49;
        bool var_50;
        bool var_51;
        const wp::float64 var_52 = 1.0;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 3;
        wp::int32 var_55;
        wp::int32 var_56;
        bool var_57;
        const wp::int32 var_58 = 0;
        const wp::int32 var_59 = 1;
        const wp::int32 var_60 = 2;
        wp::int32 var_61;
        const wp::int32 var_62 = 3;
        wp::int32 var_63;
        wp::int32 var_64;
        bool var_65;
        const wp::int32 var_66 = 0;
        const wp::int32 var_67 = 1;
        const wp::int32 var_68 = 2;
        wp::int32 var_69;
        bool var_70;
        bool var_71;
        bool var_72;
        const wp::int32 var_73 = 2;
        const wp::int32 var_74 = 3;
        wp::int32 var_75;
        wp::int32 var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        const wp::int32 var_79 = 1;
        const wp::int32 var_80 = 2;
        wp::int32 var_81;
        const wp::int32 var_82 = 3;
        wp::int32 var_83;
        wp::int32 var_84;
        bool var_85;
        const wp::int32 var_86 = 0;
        const wp::int32 var_87 = 1;
        const wp::int32 var_88 = 2;
        wp::int32 var_89;
        bool var_90;
        bool var_91;
        bool var_92;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3,wp::float64> adj_1 = {};
        wp::vec_t<3,wp::float64> adj_2 = {};
        wp::vec_t<3,wp::float64> adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::float64 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        bool adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::range_t adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::mat_t<3,3,wp::float64> adj_31 = {};
        wp::mat_t<3,3,wp::float64> adj_32 = {};
        wp::mat_t<3,3,wp::float64> adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        bool adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        bool adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        bool adj_49 = {};
        bool adj_50 = {};
        bool adj_51 = {};
        wp::float64 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        bool adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        bool adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        bool adj_70 = {};
        bool adj_71 = {};
        bool adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        bool adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::int32 adj_84 = {};
        bool adj_85 = {};
        wp::int32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        bool adj_90 = {};
        bool adj_91 = {};
        bool adj_92 = {};
        //---------
        // forward
        // def linear_constraint_project_system_kernel(                                           <L 7>
        // block_row_idx = wp.tid()                                                               <L 17>
        var_0 = builtin_tid1d();
        // grad_seg = gradient[block_row_idx]                                                     <L 18>
        var_1 = wp::address(var_gradient, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // for d_row in range(3):                                                                 <L 19>
        // row_idx = block_row_idx * 3 + d_row                                                    <L 20>
        var_6 = wp::mul(var_0, var_5);
        var_7 = wp::add(var_6, var_4);
        // if row_idx < n_reduced_dofs:                                                           <L 21>
        var_8 = (var_7 < var_n_reduced_dofs);
        if (var_8) {
            // grad_seg[d_row] = _0                                                               <L 22>
            wp::assign_inplace(var_2, var_4, var_9);
        }
        // row_idx = block_row_idx * 3 + d_row                                                    <L 20>
        var_12 = wp::mul(var_0, var_11);
        var_13 = wp::add(var_12, var_10);
        // if row_idx < n_reduced_dofs:                                                           <L 21>
        var_14 = (var_13 < var_n_reduced_dofs);
        if (var_14) {
            // grad_seg[d_row] = _0                                                               <L 22>
            wp::assign_inplace(var_2, var_10, var_9);
        }
        // row_idx = block_row_idx * 3 + d_row                                                    <L 20>
        var_17 = wp::mul(var_0, var_16);
        var_18 = wp::add(var_17, var_15);
        // if row_idx < n_reduced_dofs:                                                           <L 21>
        var_19 = (var_18 < var_n_reduced_dofs);
        if (var_19) {
            // grad_seg[d_row] = _0                                                               <L 22>
            wp::assign_inplace(var_2, var_15, var_9);
        }
        // gradient[block_row_idx] = grad_seg                                                     <L 23>
        // wp::array_store(var_gradient, var_0, var_2);
        // for k in range(offsets[block_row_idx], offsets[block_row_idx + 1]):                    <L 25>
        var_20 = wp::address(var_offsets, var_0);
        var_22 = wp::add(var_0, var_21);
        var_23 = wp::address(var_offsets, var_22);
        var_25 = wp::load(var_20);
        var_26 = wp::load(var_23);
        var_24 = wp::range(var_25, var_26);
        //---------
        // reverse
        var_24 = wp::iter_reverse(var_24);
        start_for_0:;
            if (iter_cmp(var_24) == 0) goto end_for_0;
            var_27 = wp::iter_next(var_24);
        	adj_28 = {};
        	adj_29 = {};
        	adj_30 = {};
        	adj_31 = {};
        	adj_32 = {};
        	adj_33 = {};
        	adj_34 = {};
        	adj_35 = {};
        	adj_36 = {};
        	adj_37 = {};
        	adj_38 = {};
        	adj_39 = {};
        	adj_40 = {};
        	adj_41 = {};
        	adj_42 = {};
        	adj_43 = {};
        	adj_44 = {};
        	adj_45 = {};
        	adj_46 = {};
        	adj_47 = {};
        	adj_48 = {};
        	adj_49 = {};
        	adj_50 = {};
        	adj_51 = {};
        	adj_52 = {};
        	adj_53 = {};
        	adj_54 = {};
        	adj_55 = {};
        	adj_56 = {};
        	adj_57 = {};
        	adj_58 = {};
        	adj_59 = {};
        	adj_60 = {};
        	adj_61 = {};
        	adj_62 = {};
        	adj_63 = {};
        	adj_64 = {};
        	adj_65 = {};
        	adj_66 = {};
        	adj_67 = {};
        	adj_68 = {};
        	adj_69 = {};
        	adj_70 = {};
        	adj_71 = {};
        	adj_72 = {};
        	adj_73 = {};
        	adj_74 = {};
        	adj_75 = {};
        	adj_76 = {};
        	adj_77 = {};
        	adj_78 = {};
        	adj_79 = {};
        	adj_80 = {};
        	adj_81 = {};
        	adj_82 = {};
        	adj_83 = {};
        	adj_84 = {};
        	adj_85 = {};
        	adj_86 = {};
        	adj_87 = {};
        	adj_88 = {};
        	adj_89 = {};
        	adj_90 = {};
        	adj_91 = {};
        	adj_92 = {};
            // block_col_idx = columns[k]                                                         <L 26>
            var_28 = wp::address(var_columns, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // block_val = values[k]                                                              <L 27>
            var_31 = wp::address(var_values, var_27);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // for d in range(3):                                                                 <L 28>
            // row_idx = block_row_idx * 3 + d                                                    <L 29>
            var_36 = wp::mul(var_0, var_35);
            var_37 = wp::add(var_36, var_34);
            // if row_idx < n_reduced_dofs:                                                       <L 30>
            var_38 = (var_37 < var_n_reduced_dofs);
            if (var_38) {
                // for dj in range(3):                                                            <L 31>
                // block_val[d, dj] = _0                                                          <L 32>
                wp::assign_inplace(var_32, var_34, var_39, var_9);
                wp::assign_inplace(var_32, var_34, var_40, var_9);
                wp::assign_inplace(var_32, var_34, var_41, var_9);
            }
            // col_idx = block_col_idx * 3 + d                                                    <L 34>
            var_43 = wp::mul(var_29, var_42);
            var_44 = wp::add(var_43, var_34);
            // if col_idx < n_reduced_dofs:                                                       <L 35>
            var_45 = (var_44 < var_n_reduced_dofs);
            if (var_45) {
                // for di in range(3):                                                            <L 36>
                // block_val[di, d] = _0                                                          <L 37>
                wp::assign_inplace(var_32, var_46, var_34, var_9);
                wp::assign_inplace(var_32, var_47, var_34, var_9);
                wp::assign_inplace(var_32, var_48, var_34, var_9);
            }
            // if row_idx < n_reduced_dofs and row_idx == col_idx:                                <L 39>
            var_49 = (var_37 < var_n_reduced_dofs);
            var_50 = (var_37 == var_44);
            var_51 = var_49 && var_50;
            if (var_51) {
                // block_val[d, d] = _1                                                           <L 40>
                wp::assign_inplace(var_32, var_34, var_34, var_52);
            }
            // row_idx = block_row_idx * 3 + d                                                    <L 29>
            var_55 = wp::mul(var_0, var_54);
            var_56 = wp::add(var_55, var_53);
            // if row_idx < n_reduced_dofs:                                                       <L 30>
            var_57 = (var_56 < var_n_reduced_dofs);
            if (var_57) {
                // for dj in range(3):                                                            <L 31>
                // block_val[d, dj] = _0                                                          <L 32>
                wp::assign_inplace(var_32, var_53, var_58, var_9);
                wp::assign_inplace(var_32, var_53, var_59, var_9);
                wp::assign_inplace(var_32, var_53, var_60, var_9);
            }
            var_61 = wp::where(var_57, var_60, var_41);
            // col_idx = block_col_idx * 3 + d                                                    <L 34>
            var_63 = wp::mul(var_29, var_62);
            var_64 = wp::add(var_63, var_53);
            // if col_idx < n_reduced_dofs:                                                       <L 35>
            var_65 = (var_64 < var_n_reduced_dofs);
            if (var_65) {
                // for di in range(3):                                                            <L 36>
                // block_val[di, d] = _0                                                          <L 37>
                wp::assign_inplace(var_32, var_66, var_53, var_9);
                wp::assign_inplace(var_32, var_67, var_53, var_9);
                wp::assign_inplace(var_32, var_68, var_53, var_9);
            }
            var_69 = wp::where(var_65, var_68, var_48);
            // if row_idx < n_reduced_dofs and row_idx == col_idx:                                <L 39>
            var_70 = (var_56 < var_n_reduced_dofs);
            var_71 = (var_56 == var_64);
            var_72 = var_70 && var_71;
            if (var_72) {
                // block_val[d, d] = _1                                                           <L 40>
                wp::assign_inplace(var_32, var_53, var_53, var_52);
            }
            // row_idx = block_row_idx * 3 + d                                                    <L 29>
            var_75 = wp::mul(var_0, var_74);
            var_76 = wp::add(var_75, var_73);
            // if row_idx < n_reduced_dofs:                                                       <L 30>
            var_77 = (var_76 < var_n_reduced_dofs);
            if (var_77) {
                // for dj in range(3):                                                            <L 31>
                // block_val[d, dj] = _0                                                          <L 32>
                wp::assign_inplace(var_32, var_73, var_78, var_9);
                wp::assign_inplace(var_32, var_73, var_79, var_9);
                wp::assign_inplace(var_32, var_73, var_80, var_9);
            }
            var_81 = wp::where(var_77, var_80, var_61);
            // col_idx = block_col_idx * 3 + d                                                    <L 34>
            var_83 = wp::mul(var_29, var_82);
            var_84 = wp::add(var_83, var_73);
            // if col_idx < n_reduced_dofs:                                                       <L 35>
            var_85 = (var_84 < var_n_reduced_dofs);
            if (var_85) {
                // for di in range(3):                                                            <L 36>
                // block_val[di, d] = _0                                                          <L 37>
                wp::assign_inplace(var_32, var_86, var_73, var_9);
                wp::assign_inplace(var_32, var_87, var_73, var_9);
                wp::assign_inplace(var_32, var_88, var_73, var_9);
            }
            var_89 = wp::where(var_85, var_88, var_69);
            // if row_idx < n_reduced_dofs and row_idx == col_idx:                                <L 39>
            var_90 = (var_76 < var_n_reduced_dofs);
            var_91 = (var_76 == var_84);
            var_92 = var_90 && var_91;
            if (var_92) {
                // block_val[d, d] = _1                                                           <L 40>
                wp::assign_inplace(var_32, var_73, var_73, var_52);
            }
            // values[k] = block_val                                                              <L 41>
            // wp::array_store(var_values, var_27, var_32);
            wp::assign(var_18, var_76);
            wp::adj_assign(var_18, var_76, adj_18, adj_76);
            wp::adj_array_store(var_values, var_27, var_32, adj_values, adj_27, adj_32);
            // adj: values[k] = block_val                                                         <L 41>
            if (var_92) {
                wp::adj_assign_inplace(var_32, var_73, var_73, var_52, adj_32, adj_73, adj_73, adj_52);
                // adj: block_val[d, d] = _1                                                      <L 40>
            }
            // adj: if row_idx < n_reduced_dofs and row_idx == col_idx:                           <L 39>
            wp::adj_where(var_85, var_88, var_69, adj_85, adj_88, adj_69, adj_89);
            if (var_85) {
                wp::adj_assign_inplace(var_32, var_88, var_73, var_9, adj_32, adj_88, adj_73, adj_9);
                wp::adj_assign_inplace(var_32, var_87, var_73, var_9, adj_32, adj_87, adj_73, adj_9);
                wp::adj_assign_inplace(var_32, var_86, var_73, var_9, adj_32, adj_86, adj_73, adj_9);
                // adj: block_val[di, d] = _0                                                     <L 37>
                // adj: for di in range(3):                                                       <L 36>
            }
            // adj: if col_idx < n_reduced_dofs:                                                  <L 35>
            wp::adj_add(var_83, var_73, adj_83, adj_73, adj_84);
            wp::adj_mul(var_29, var_82, adj_29, adj_82, adj_83);
            // adj: col_idx = block_col_idx * 3 + d                                               <L 34>
            wp::adj_where(var_77, var_80, var_61, adj_77, adj_80, adj_61, adj_81);
            if (var_77) {
                wp::adj_assign_inplace(var_32, var_73, var_80, var_9, adj_32, adj_73, adj_80, adj_9);
                wp::adj_assign_inplace(var_32, var_73, var_79, var_9, adj_32, adj_73, adj_79, adj_9);
                wp::adj_assign_inplace(var_32, var_73, var_78, var_9, adj_32, adj_73, adj_78, adj_9);
                // adj: block_val[d, dj] = _0                                                     <L 32>
                // adj: for dj in range(3):                                                       <L 31>
            }
            // adj: if row_idx < n_reduced_dofs:                                                  <L 30>
            wp::adj_add(var_75, var_73, adj_75, adj_73, adj_76);
            wp::adj_mul(var_0, var_74, adj_0, adj_74, adj_75);
            // adj: row_idx = block_row_idx * 3 + d                                               <L 29>
            if (var_72) {
                wp::adj_assign_inplace(var_32, var_53, var_53, var_52, adj_32, adj_53, adj_53, adj_52);
                // adj: block_val[d, d] = _1                                                      <L 40>
            }
            // adj: if row_idx < n_reduced_dofs and row_idx == col_idx:                           <L 39>
            wp::adj_where(var_65, var_68, var_48, adj_65, adj_68, adj_48, adj_69);
            if (var_65) {
                wp::adj_assign_inplace(var_32, var_68, var_53, var_9, adj_32, adj_68, adj_53, adj_9);
                wp::adj_assign_inplace(var_32, var_67, var_53, var_9, adj_32, adj_67, adj_53, adj_9);
                wp::adj_assign_inplace(var_32, var_66, var_53, var_9, adj_32, adj_66, adj_53, adj_9);
                // adj: block_val[di, d] = _0                                                     <L 37>
                // adj: for di in range(3):                                                       <L 36>
            }
            // adj: if col_idx < n_reduced_dofs:                                                  <L 35>
            wp::adj_add(var_63, var_53, adj_63, adj_53, adj_64);
            wp::adj_mul(var_29, var_62, adj_29, adj_62, adj_63);
            // adj: col_idx = block_col_idx * 3 + d                                               <L 34>
            wp::adj_where(var_57, var_60, var_41, adj_57, adj_60, adj_41, adj_61);
            if (var_57) {
                wp::adj_assign_inplace(var_32, var_53, var_60, var_9, adj_32, adj_53, adj_60, adj_9);
                wp::adj_assign_inplace(var_32, var_53, var_59, var_9, adj_32, adj_53, adj_59, adj_9);
                wp::adj_assign_inplace(var_32, var_53, var_58, var_9, adj_32, adj_53, adj_58, adj_9);
                // adj: block_val[d, dj] = _0                                                     <L 32>
                // adj: for dj in range(3):                                                       <L 31>
            }
            // adj: if row_idx < n_reduced_dofs:                                                  <L 30>
            wp::adj_add(var_55, var_53, adj_55, adj_53, adj_56);
            wp::adj_mul(var_0, var_54, adj_0, adj_54, adj_55);
            // adj: row_idx = block_row_idx * 3 + d                                               <L 29>
            if (var_51) {
                wp::adj_assign_inplace(var_32, var_34, var_34, var_52, adj_32, adj_34, adj_34, adj_52);
                // adj: block_val[d, d] = _1                                                      <L 40>
            }
            // adj: if row_idx < n_reduced_dofs and row_idx == col_idx:                           <L 39>
            if (var_45) {
                wp::adj_assign_inplace(var_32, var_48, var_34, var_9, adj_32, adj_48, adj_34, adj_9);
                wp::adj_assign_inplace(var_32, var_47, var_34, var_9, adj_32, adj_47, adj_34, adj_9);
                wp::adj_assign_inplace(var_32, var_46, var_34, var_9, adj_32, adj_46, adj_34, adj_9);
                // adj: block_val[di, d] = _0                                                     <L 37>
                // adj: for di in range(3):                                                       <L 36>
            }
            // adj: if col_idx < n_reduced_dofs:                                                  <L 35>
            wp::adj_add(var_43, var_34, adj_43, adj_34, adj_44);
            wp::adj_mul(var_29, var_42, adj_29, adj_42, adj_43);
            // adj: col_idx = block_col_idx * 3 + d                                               <L 34>
            if (var_38) {
                wp::adj_assign_inplace(var_32, var_34, var_41, var_9, adj_32, adj_34, adj_41, adj_9);
                wp::adj_assign_inplace(var_32, var_34, var_40, var_9, adj_32, adj_34, adj_40, adj_9);
                wp::adj_assign_inplace(var_32, var_34, var_39, var_9, adj_32, adj_34, adj_39, adj_9);
                // adj: block_val[d, dj] = _0                                                     <L 32>
                // adj: for dj in range(3):                                                       <L 31>
            }
            // adj: if row_idx < n_reduced_dofs:                                                  <L 30>
            wp::adj_add(var_36, var_34, adj_36, adj_34, adj_37);
            wp::adj_mul(var_0, var_35, adj_0, adj_35, adj_36);
            // adj: row_idx = block_row_idx * 3 + d                                               <L 29>
            // adj: for d in range(3):                                                            <L 28>
            wp::adj_copy(var_33, adj_31, adj_32);
            wp::adj_load(var_31, adj_31, adj_33);
            wp::adj_address(var_values, var_27, adj_values, adj_27, adj_31);
            // adj: block_val = values[k]                                                         <L 27>
            wp::adj_copy(var_30, adj_28, adj_29);
            wp::adj_load(var_28, adj_28, adj_30);
            wp::adj_address(var_columns, var_27, adj_columns, adj_27, adj_28);
            // adj: block_col_idx = columns[k]                                                    <L 26>
        	goto start_for_0;
        end_for_0:;
        wp::adj_range(var_25, var_26, adj_20, adj_23, adj_24);
        wp::adj_load(var_23, adj_23, adj_26);
        wp::adj_load(var_20, adj_20, adj_25);
        wp::adj_address(var_offsets, var_22, adj_offsets, adj_22, adj_23);
        wp::adj_add(var_0, var_21, adj_0, adj_21, adj_22);
        wp::adj_address(var_offsets, var_0, adj_offsets, adj_0, adj_20);
        // adj: for k in range(offsets[block_row_idx], offsets[block_row_idx + 1]):               <L 25>
        wp::adj_array_store(var_gradient, var_0, var_2, adj_gradient, adj_0, adj_2);
        // adj: gradient[block_row_idx] = grad_seg                                                <L 23>
        if (var_19) {
            wp::adj_assign_inplace(var_2, var_15, var_9, adj_2, adj_15, adj_9);
            // adj: grad_seg[d_row] = _0                                                          <L 22>
        }
        // adj: if row_idx < n_reduced_dofs:                                                      <L 21>
        wp::adj_add(var_17, var_15, adj_17, adj_15, adj_18);
        wp::adj_mul(var_0, var_16, adj_0, adj_16, adj_17);
        // adj: row_idx = block_row_idx * 3 + d_row                                               <L 20>
        if (var_14) {
            wp::adj_assign_inplace(var_2, var_10, var_9, adj_2, adj_10, adj_9);
            // adj: grad_seg[d_row] = _0                                                          <L 22>
        }
        // adj: if row_idx < n_reduced_dofs:                                                      <L 21>
        wp::adj_add(var_12, var_10, adj_12, adj_10, adj_13);
        wp::adj_mul(var_0, var_11, adj_0, adj_11, adj_12);
        // adj: row_idx = block_row_idx * 3 + d_row                                               <L 20>
        if (var_8) {
            wp::adj_assign_inplace(var_2, var_4, var_9, adj_2, adj_4, adj_9);
            // adj: grad_seg[d_row] = _0                                                          <L 22>
        }
        // adj: if row_idx < n_reduced_dofs:                                                      <L 21>
        wp::adj_add(var_6, var_4, adj_6, adj_4, adj_7);
        wp::adj_mul(var_0, var_5, adj_0, adj_5, adj_6);
        // adj: row_idx = block_row_idx * 3 + d_row                                               <L 20>
        // adj: for d_row in range(3):                                                            <L 19>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_gradient, var_0, adj_gradient, adj_0, adj_1);
        // adj: grad_seg = gradient[block_row_idx]                                                <L 18>
        // adj: block_row_idx = wp.tid()                                                          <L 17>
        // adj: def linear_constraint_project_system_kernel(                                      <L 7>
        continue;
    }
}

