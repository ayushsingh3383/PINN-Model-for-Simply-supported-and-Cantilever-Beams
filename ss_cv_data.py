import pandas as pd
import numpy as np
from pathlib import Path

def ss_data(L_physical,q_physical,E,I,N=100):
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "data" / "ss_data_1.xlsx"

    df_1=pd.read_excel(file_path)

    x_test_norm_np=np.linspace(0,1,N)
    x_scale=L_physical

    x_phys=x_test_norm_np*x_scale
    d_truth_phy = (q_physical/(24*E*I))*(x_phys**4 - 2*L_physical*x_phys**3 + L_physical**3*x_phys)
    df_1['deflection(y)']=d_truth_phy
    df_1['position(X)']=x_phys
    df_1.to_excel(file_path, index=False)
# ss_data(L_physical=8.0,q_physical=10.0,E=210e9,I=1/12)

def cv_data(L_physical,q_physical,E,I,N=100):
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "data" / "cv_data_1.xlsx"
    df_2=pd.read_excel(file_path)
    x_test_norm_np=np.linspace(0,1,N)
    x_scale=L_physical

    x_phys=x_test_norm_np*x_scale
    d_truth_phy = (q_physical/(24*E*I))*(x_phys**4 - 4*L_physical*x_phys**3 + 6*L_physical**2*x_phys**2)
    df_2['deflection(y)']=d_truth_phy
    df_2['position(X)']=x_phys
    df_2.to_excel(file_path, index=False)
# cv_data(L_physical=8.0,q_physical=10.0,E=210e9,I=1/12)