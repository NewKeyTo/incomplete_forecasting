def deNorm(x, std, mean):
    return (x * std) + mean

def mixedMSELoss(predict, gd, mask, mask_):
    error = (predict - gd) ** 2
    r = (error * (mask - mask_)).sum() / (mask - mask_).sum()
    r += (error * mask_).sum() / mask_.sum()
    return r

def mixedAbsLoss(predict, gd, mask, mask_):
    error = (predict - gd).abs()
    r = (error * (mask - mask_)).sum() / (mask - mask_).sum()
    r += (error * mask_).sum() / mask_.sum()
    return r

def fullinput_Valid_MSELoss(predict, gd, mask):
    error = (predict - gd) * mask
    r = (error ** 2).sum() / mask.sum()
    return r

def fullinput_Valid_AbsLoss(predict, gd, mask):
    error = (predict - gd) * mask
    r = error.abs().sum() / mask.sum()
    return r

def fullinput_Missing_MSELoss(predict, gd, mask):
    error = (predict - gd) * (1-mask)
    r = (error ** 2).sum() / (1-mask).sum()
    return r

def fullinput_Missing_AbsLoss(predict, gd, mask):
    error = (predict - gd) * (1-mask)
    r = error.abs().sum() / (1-mask).sum()
    return r

def maskinput_AM_MSELoss(predict, gd, mask, mask_):
    error = (predict - gd) * (mask - mask_)
    r = (error ** 2).sum() / (mask - mask_).sum()
    return r

def maskinput_Valid_MSELoss(predict, gd, mask_):
    error = (predict - gd) * mask_
    r = (error ** 2).sum() / mask_.sum()
    return r

def maskinput_Missing_MSELoss(predict, gd, mask):
    error = (predict - gd) * (1-mask)
    r = (error ** 2).sum() / (1-mask + 1e-7).sum()
    return r

def maskinput_AM_AbsLoss(predict, gd, mask, mask_):
    error = (predict - gd) * (mask - mask_)
    r = error.abs().sum() / (mask - mask_).sum()
    return r

def maskinput_Valid_AbsLoss(predict, gd, mask_):
    error = (predict - gd) * mask_
    r = error.abs().sum() / mask_.sum()
    return r

def maskinput_Missing_AbsLoss(predict, gd, mask):
    error = (predict - gd) * (1-mask)
    r = error.abs().sum() / (1-mask).sum()
    return r
