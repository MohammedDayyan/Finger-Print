from ml.inference import predict_fingerprint

if __name__ == "__main__":
    result = predict_fingerprint("SOCOFing/Altered/Altered-Easy/302__M_Right_index_finger_CR.BMP")
    print(result)
