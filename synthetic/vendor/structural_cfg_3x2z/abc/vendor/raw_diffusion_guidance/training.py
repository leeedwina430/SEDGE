import torch

def calibrate(model,vx,vz):
    device=next(model.parameters()).device
    with torch.no_grad():residual=torch.cat([(vz[i:i+512].to(device)-model(vx[i:i+512].to(device))).cpu() for i in range(0,len(vx),512)]).double()
    b=residual.mean(0);r=residual-b;k=r.shape[1];cov=r.T@r/len(r)
    floor=max(1e-4,1e-3*float(torch.trace(cov))/k)
    sigma=.95*cov+.05*torch.diag(torch.diag(cov))+floor*torch.eye(k,dtype=torch.float64)
    eigen=torch.linalg.eigvalsh(sigma)
    return dict(bias=b.float(),covariance=sigma.float(),cholesky=torch.linalg.cholesky(sigma.float()),
        sample_covariance=cov,calibration_rows=len(r),calibration_split='selected_validation',shrinkage=.05,
        ridge_floor=floor,covariance_eigenvalues=eigen,residual_mse=residual.square().mean(0),
        fitted_statistics=k+k*(k+1)//2)

