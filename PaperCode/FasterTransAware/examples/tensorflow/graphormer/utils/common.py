import numpy as np

def cross_check(name, tf_val, op_val, atol_threshold):
  abs_diff = np.fabs(tf_val - op_val)
  print("[INFO] {} Cross check {}".format(name, np.allclose(tf_val, op_val, atol=atol_threshold)))
  print("[INFO] Max diff {}".format(abs_diff.max()))
  print("[INFO] min diff {}".format(abs_diff.min()))