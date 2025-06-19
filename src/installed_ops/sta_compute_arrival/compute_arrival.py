import torch
import installed_ops.sta_compute_arrival.sta_compute_arrival_cuda as compute_arrival


class ComputeArrivalPOCV():
    @staticmethod
    def forward(p_rise_means, p_rise_stds, p_rise_sps,
                p_fall_means, p_fall_stds, p_fall_sps,
                c_rise_means, c_rise_stds, c_rise_sigmas,
                c_fall_means, c_fall_stds, c_fall_sigmas,
                duplicated_senses, node_start_idx, sigmas,
                p_indices, p_mapping,
                valid_sps,
                topK,
                float_dtype):

        if float_dtype == torch.float32:
            (rise_means, rise_stds, rise_sps,
             fall_means, fall_stds, fall_sps ) = \
                     compute_arrival.compute_rise_fall_arrival_pocv(
                             p_rise_means, p_rise_stds, p_rise_sps,
                             p_fall_means, p_fall_stds, p_fall_sps,
                             c_rise_means, c_rise_stds, c_rise_sigmas,
                             c_fall_means, c_fall_stds, c_fall_sigmas,
                             duplicated_senses, node_start_idx, sigmas,
                             p_indices, p_mapping,
                             valid_sps,
                             topK)
        elif float_dtype == torch.float16 or float_dtype == torch.bfloat16:
            (rise_means, rise_stds, rise_sps,
             fall_means, fall_stds, fall_sps ) = \
                     compute_arrival.compute_rise_fall_arrival_pocv_bfloat16(
                             p_rise_means, p_rise_stds, p_rise_sps,
                             p_fall_means, p_fall_stds, p_fall_sps,
                             c_rise_means, c_rise_stds, c_rise_sigmas,
                             c_fall_means, c_fall_stds, c_fall_sigmas,
                             duplicated_senses, node_start_idx, sigmas,
                             p_indices, p_mapping,
                             valid_sps,
                             topK)


        torch.cuda.synchronize()
        return (rise_means, rise_stds, rise_sps,
                fall_means, fall_stds, fall_sps )


class ComputeArrivalPOCVWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                p_rise_means, p_rise_stds, p_rise_sps,
                p_fall_means, p_fall_stds, p_fall_sps,
                c_rise_means, c_rise_stds, c_rise_sigmas,
                c_fall_means, c_fall_stds, c_fall_sigmas,
                duplicated_senses, node_start_idx, sigmas,
                p_indices, p_mapping,
                temperatures):

        (rise_means, rise_stds, rise_sps, rise_sigmas,
         fall_means, fall_stds, fall_sps, fall_sigmas,
         rise_mean_grads, fall_mean_grads) = \
                 compute_arrival.compute_rise_fall_arrival_pocv_with_grad(
                         p_rise_means, p_rise_stds, p_rise_sps,
                         p_fall_means, p_fall_stds, p_fall_sps,
                         c_rise_means, c_rise_stds, c_rise_sigmas,
                         c_fall_means, c_fall_stds, c_fall_sigmas,
                         duplicated_senses, node_start_idx, sigmas,
                         p_indices, p_mapping,
                         temperatures)

        #print('checking gradient...')
        #assert not torch.isinf(rise_mean_grads).any(), rise_mean_grads
        #assert not torch.isnan(rise_mean_grads).any(), rise_mean_grads
        #assert not torch.isinf(fall_mean_grads).any() and not torch.isnan(fall_mean_grads).any(), fall_mean_grads

        ctx.p_mapping = p_mapping
        ctx.p_indices = p_indices
        ctx.senses = duplicated_senses
        ctx.node_start_idx = node_start_idx
        ctx.rise_mean_grads = rise_mean_grads
        ctx.fall_mean_grads = fall_mean_grads

        #print(f'rise_mean_grad max: {rise_mean_grads.max()}, min: {rise_mean_grads.min()}')
        #print(f'fall_mean_grad max: {fall_mean_grads.max()}, min: {fall_mean_grads.min()}')

        torch.cuda.synchronize()
        return (rise_means, rise_stds, rise_sps, rise_sigmas,
                fall_means, fall_stds, fall_sps, fall_sigmas)


    @staticmethod
    def backward(ctx,
                dL_dRiseMeans, dL_dRiseStds, dL_dRiseSps, dL_dRiseSigmas,
                dL_dFallMeans, dL_dFallStds, dL_dFallSps, dL_dFallSigmas
        ):
        #assert not torch.isinf(dL_dRiseMeans).any() and not torch.isnan(dL_dRiseMeans).any(), f"dL_dRiseMeans: {dL_dRiseMeans}"
        #assert not torch.isinf(dL_dFallMeans).any() and not torch.isnan(dL_dFallMeans).any(), f"dL_dFallMeans: {dL_dFallMeans}"

        #print(f'dL_dRiseMeans max: {dL_dRiseMeans.max()}, min: {dL_dRiseMeans.min()}')
        #print(f'dL_dFallMeans max: {dL_dFallMeans.max()}, min: {dL_dFallMeans.min()}')
        #print(f'[debug] dL_dRiseMeans sum: {dL_dRiseMeans.sum()}, dL_dFallMeans sum: {dL_dFallMeans.sum()}, total sum: {dL_dRiseMeans.sum() + dL_dFallMeans.sum()}')

        dL_dpRiseMeans, dL_dpFallMeans, dL_dcRiseMeans, dL_dcFallMeans =\
            compute_arrival.compute_rise_fall_arrival_pocv_with_grad_bw(
                ctx.senses, ctx.node_start_idx,
                ctx.p_indices, ctx.p_mapping,
                dL_dRiseMeans, dL_dFallMeans,
                ctx.rise_mean_grads, ctx.fall_mean_grads
            )
        #assert not torch.isinf(dL_dpRiseMeans).any() and not torch.isnan(dL_dpRiseMeans).any(), f"dL_dpRiseMeans: {dL_dpRiseMeans}"
        #assert not torch.isinf(dL_dpFallMeans).any() and not torch.isnan(dL_dpFallMeans).any(), f"dL_dpFallMeans: {dL_dpFallMeans}"
        #assert not torch.isinf(dL_dcRiseMeans).any() and not torch.isnan(dL_dcRiseMeans).any(), f"dL_dcRiseMeans: {dL_dcRiseMeans}"
        #assert not torch.isinf(dL_dcFallMeans).any() and not torch.isnan(dL_dcFallMeans).any(), f"dL_dcFallMeans: {dL_dcFallMeans}"

        return (dL_dpRiseMeans, None, None,
                dL_dpFallMeans, None, None,
                dL_dcRiseMeans, None, None,
                dL_dcFallMeans, None, None,
                None, None, None,
                None, None, None,
                None, None, None,
                None, None,
                None)


'''
class ComputeArrivalPOCVWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                p_rise_means, p_rise_stds, p_rise_sps,
                p_fall_means, p_fall_stds, p_fall_sps,
                c_rise_means, c_rise_stds, c_rise_sigmas,
                c_fall_means, c_fall_stds, c_fall_sigmas,
                duplicated_senses, node_start_idx, sigmas,
                p_indices, p_mapping,
                temperatures):

        (rise_means, rise_stds, rise_sps, rise_sigmas,
         fall_means, fall_stds, fall_sps, fall_sigmas,
         rise_mean_grads, fall_mean_grads) = \
                 compute_arrival.compute_rise_fall_arrival_pocv_with_grad(
                         p_rise_means, p_rise_stds, p_rise_sps,
                         p_fall_means, p_fall_stds, p_fall_sps,
                         c_rise_means, c_rise_stds, c_rise_sigmas,
                         c_fall_means, c_fall_stds, c_fall_sigmas,
                         duplicated_senses, node_start_idx, sigmas,
                         p_indices, p_mapping,
                         temperatures)

        #print('checking gradient...')
        assert not torch.isinf(rise_mean_grads).any() and not torch.isnan(rise_mean_grads).any(), rise_mean_grads
        assert not torch.isinf(fall_mean_grads).any() and not torch.isnan(fall_mean_grads).any(), fall_mean_grads

        ctx.p_mapping = p_mapping
        ctx.p_indices = p_indices
        ctx.senses = duplicated_senses
        ctx.node_start_idx = node_start_idx
        ctx.rise_mean_grads = rise_mean_grads
        ctx.fall_mean_grads = fall_mean_grads

        torch.cuda.synchronize()
        return (rise_means, rise_stds, rise_sps, rise_sigmas,
                fall_means, fall_stds, fall_sps, fall_sigmas)


    @staticmethod
    def backward(ctx,
                dL_dRiseMeans, dL_dRiseStds, dL_dRiseSps, dL_dRiseSigmas,
                dL_dFallMeans, dL_dFallStds, dL_dFallSps, dL_dFallSigmas
        ):
        dL_dpRiseMeans, dL_dpFallMeans = compute_arrival.compute_rise_fall_arrival_pocv_with_grad_bw(
                ctx.senses, ctx.node_start_idx,
                ctx.p_indices, ctx.p_mapping,
                dL_dRiseMeans, dL_dFallMeans,
                ctx.rise_mean_grads, ctx.fall_mean_grads
                )
        assert not torch.isinf(dL_dpRiseMeans).any() and not torch.isnan(dL_dpRiseMeans).any(), dL_dpRiseMeans
        assert not torch.isinf(dL_dpFallMeans).any() and not torch.isnan(dL_dpFallMeans).any(), dL_dpFallMeans

        return (dL_dpRiseMeans, None, None,
                dL_dpFallMeans, None, None,
                None, None, None,
                None, None, None,
                None, None, None,
                None, None,
                None)
'''


