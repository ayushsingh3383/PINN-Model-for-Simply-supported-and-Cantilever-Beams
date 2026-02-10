from Euler import L_phsyical,q_physical,E,I,x_scale,d_scale,beam_type
from ss_cv_data import ss_data,cv_data
from Euler import PINN
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from pathlib import Path

# importing the saved model here 
device = torch.device("cpu")
model=PINN()
model.load_state_dict(torch.load("model.pth", map_location='cpu'))

# genrating normalized points on the beam
x_test_norm_np=np.linspace(0,1,100)
x_test_norm_tensor=torch.tensor(x_test_norm_np,dtype=torch.float32).view(-1,1)
model.eval()
# gradient need not to tracked during the testing
with torch.no_grad():
    d_norm=model(x_test_norm_tensor)
    d_phy=d_norm.numpy()*d_scale

x_phys=x_test_norm_np*x_scale
x_test_phys=x_phys

if beam_type=='simply':
    ss_data(L_physical=L_phsyical,q_physical=q_physical,E=E,I=I)
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "data" / "ss_data_1.xlsx"
    try:
        print("Loading data from 'ss_data_1.xlsv'...")
        data=pd.read_excel(file_path)
        deflection_phy=data['deflection(y)'].to_numpy()
        position_phy=data['position(X)'].to_numpy()
        print("FEA data loaded successfully.")
    except EOFError:
        print("--- ERROR ---")
        print("'ss_data_1.dat' file not found.")
        print("Please save your data to that file in the same directory.")
        # Create empty arrays so the plot still runs (but shows no FEA)
        deflection_phy = np.array([])
        position_phy = np.array([])
    
    plt.figure(figsize=(12, 8))
    # Plot the PINN Prediction as a dashed red line
    plt.plot(x_test_phys, d_phy, 'o--', label=f'PINN Prediction', linewidth=1)
    
    # Plot the Calculix (FEA) data as blue circles
    plt.plot(position_phy, deflection_phy, 'r--', label='Calculix (FEA)', markersize=1)
    plt.title('PINN vs. FEA: Beam Deflection', fontsize=18)
    plt.xlabel('Position along beam (x) in meters', fontsize=14)
    plt.ylabel('Vertical Deflection (d) in meters', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Format the y-axis to use scientific notation (since numbers are small)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.show()
    
else:
    cv_data(L_physical=L_phsyical,q_physical=q_physical,E=E,I=I)
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "data" / "ss_data_1.xlsx"
    try:
        print("Loading data from 'cv_data_1.xlsv'...")
        data=pd.read_excel(file_path)
        deflection_phy=data['deflection(y)'].to_numpy()
        position_phy=data['position(X)'].to_numpy()
        print("FEA data loaded successfully.")
    except EOFError:
        print("--- ERROR ---")
        print("'cv_data_1.dat' file not found.")
        print("Please save your data to that file in the same directory.")
        # Create empty arrays so the plot still runs (but shows no FEA)
        deflection_phy = np.array([])
        position_phy = np.array([])
    
    plt.figure(figsize=(12, 8))
    # Plot the PINN Prediction as a dashed red line
    plt.plot(x_test_phys, d_phy, 'o--', label=f'PINN Prediction', linewidth=1)
    
    # Plot the Calculix (FEA) data as blue circles
    plt.plot(position_phy, deflection_phy, 'r--', label='Calculix (FEA)', markersize=1)
    plt.title('PINN vs. FEA: Beam Deflection', fontsize=18)
    plt.xlabel('Position along beam (x) in meters', fontsize=14)
    plt.ylabel('Vertical Deflection (d) in meters', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Format the y-axis to use scientific notation (since numbers are small)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.show()


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred)**2)

mse=mean_squared_error(deflection_phy,d_phy)
print(f'the mean square error is{mse:.2f} ')