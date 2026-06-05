def judge_status(defect_area, threshold):
    if defect_area >= threshold:
        return "NG"
    else:
        return "OK"