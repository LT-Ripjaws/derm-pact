"""Small fixed-feature corruption detector; distinct from clinical image usability."""
import numpy as np
from scipy.ndimage import laplace, sobel, gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES=('mean','std','p01','p10','p90','p99','dark_fraction','bright_fraction',
               'log_laplacian_variance','gradient_mean','gradient_p90','highfreq_energy',
               'entropy','saturation_mean','saturation_std','channel_spread','center_std',
               'border_gradient','center_gradient','horizontal_edge_ratio')


def describe(image):
    rgb=np.asarray(image.convert('RGB').resize((224,224)),dtype=float)/255
    g=rgb@np.array([.299,.587,.114]);dx=sobel(g,axis=1);dy=sobel(g,axis=0)
    grad=np.hypot(dx,dy);hist=np.histogram(g,bins=32,range=(0,1))[0].astype(float);hist/=hist.sum()
    sat=rgb.max(2)-rgb.min(2);center=g[56:168,56:168];border=np.ones(g.shape,dtype=bool);border[28:196,28:196]=False
    values=[g.mean(),g.std(),*np.quantile(g,[.01,.1,.9,.99]),(g<.03).mean(),(g>.97).mean(),
            np.log(max(np.var(laplace(g)),1e-12)),grad.mean(),np.quantile(grad,.9),
            np.mean((g-gaussian_filter(g,1))**2),-np.sum(hist*np.log(np.maximum(hist,1e-12))),
            sat.mean(),sat.std(),rgb.mean((0,1)).std(),center.std(),grad[border].mean(),
            grad[56:168,56:168].mean(),np.mean(np.abs(dx))/(np.mean(np.abs(dy))+1e-8)]
    return np.asarray(values,dtype=float)


def fit_detector(features,labels):
    x=np.asarray(features,dtype=float);y=np.asarray(labels,dtype=int)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or len(x)!=len(y) or not np.isfinite(x).all():
        raise ValueError('invalid quality features')
    if set(y)!={0,1}:raise ValueError('both clean and corruption classes required')
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=20260906).fit(scaler.transform(x),y)
    return {'feature_names':list(FEATURE_NAMES),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),
            'meaning':'probability of specified synthetic corruption, NOT clinical usability probability'}


def score_detector(features,parameters):
    from scipy.special import expit
    x=np.asarray(features,dtype=float)
    if parameters['feature_names']!=list(FEATURE_NAMES) or x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('quality feature contract mismatch')
    scale=np.asarray(parameters['scale']);coef=np.asarray(parameters['coef']);mean=np.asarray(parameters['mean'])
    if scale.shape!=mean.shape or scale.shape!=coef.shape or scale.shape!=(len(FEATURE_NAMES),) or not np.isfinite(parameters['intercept']):
        raise ValueError('invalid quality parameter shapes or intercept')
    if not np.isfinite(scale).all() or np.any(scale<=0) or not np.isfinite(coef).all() or not np.isfinite(mean).all():
        raise ValueError('invalid quality parameters')
    return expit((x-mean)/scale@coef+parameters['intercept'])
