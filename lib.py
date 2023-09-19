def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    other = size[:-2]
    feat_var = feat.reshape(*other, -1).var(dim=-1) + eps
    feat_std = feat_var.sqrt().reshape(*other, 1, 1)
    feat_mean = feat.reshape(*other, -1).mean(dim=-1).reshape(*other, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat = None, is_simplied = False, style_mean = None, style_std = None):
    size = content_feat.size()
    if not is_simplied:
        assert style_feat is not None
        assert (content_feat.size()[:-2] == style_feat.size()[:-2])
        style_mean, style_std = calc_mean_std(style_feat)
    else:
        assert style_mean is not None
        assert style_std is not None
    content_mean, content_std = calc_mean_std(content_feat)
    # return content_feat - content_mean.expand(size) + style_mean.expand(size)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def block_adaIN(content_feat, style_feat = None, blocknum = 16, is_simplied = False, style_mean = None, style_std = None):
    if not is_simplied:
        assert (content_feat.size()[:-2] == style_feat.size()[:-2])
        content_feat = blockzation(content_feat, blocknum)
        style_feat = blockzation(style_feat, blocknum)
        return  unblockzation(adaptive_instance_normalization(content_feat, style_feat))
    else:
        assert style_mean is not None
        assert style_std is not None
        content_feat = blockzation(content_feat, blocknum)
        return unblockzation(adaptive_instance_normalization(content_feat, is_simplied=True, style_mean=style_mean, style_std=style_std))

def blockzation(feat, blocknum = 16):
    H, W = feat.size()[-2:]
    assert H % blocknum == 0
    assert W % blocknum == 0
    size = feat.size()[:-2]
    feat = feat.reshape(*size,blocknum, H // blocknum, blocknum, W // blocknum).transpose(-2, -3)
    return feat

def unblockzation(feat):
    size = feat.size()
    H = size[-4] * size[-2]
    W = size[-3] * size[-1]
    size = size[:-4]
    return feat.transpose(-2, -3).reshape(*size, H, W)