class VoiceTrans(object):
    """
    音声の前処理を行う
    """
    def __init__(self, maxi, mini):
        """
        init
        入力するデータの最大値と最小値を必要とする
        """
        self.maxi = maxi
        self.mini = mini

    def norm_voice(self, array):
        """
        最大値を1, 最小値を0とする変換
        """
        array -= self.mini
        array /= self.maxi
    
        return array
    
    def cut(self, voice):
        """
        モデルの都合上
        音声の長さを4の倍数にする
        下端を少し削る
        """
        return voice[:, :voice.shape[1]-voice.shape[1]%4]
    
    def __call__(self, sample):
        """
        transした値を返す
        """
        return self.cut(self.norm_voice(sample))




class ImageTrans(object):
    """
    画像の前処理をする
    """
    def __init__(self):
        """
        init
        """
        pass

    def norm_image(self, array):
        """
        最大値を1, 最小値を0とする
        """
        return array/255
    
    def __call__(self, sample):
        """
        transした値を返す
        """
        return self.norm_image(sample).T