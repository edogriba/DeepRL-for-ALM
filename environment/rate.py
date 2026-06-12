from environment.config import BETA0_GRID, BETA1_GRID, BETA2
from models.utils import get_nelson_siegel_yield

def __main__():
    print("Testing yield curve generation...")
    for i0 in range(len(BETA0_GRID)):
        for i1 in range(len(BETA1_GRID)):
            beta = [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]
            y1m = get_nelson_siegel_yield(1.0 / 12.0, beta)
            y2m = get_nelson_siegel_yield(2.0 / 12.0, beta)
            y3m = get_nelson_siegel_yield(3.0 / 12.0, beta)
            y6m = get_nelson_siegel_yield(6.0 / 12.0, beta)
            print(f"Beta: {beta} | 1M Yield: {y1m:.4f} | 2M Yield: {y2m:.4f}"+
                  f" | 3M Yield: {y3m:.4f} | 6M Yield: {y6m:.4f}")

    print("Copia e incolla il seguente output su Overleaf dentro l'ambiente \\tabular:\n")
    
    for i0 in range(len(BETA0_GRID)):
        for i1 in range(len(BETA1_GRID)):
            beta = [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]
            
            y1m = get_nelson_siegel_yield(1.0 / 12.0, beta)
            y3m = get_nelson_siegel_yield(3.0 / 12.0, beta)
            y6m = get_nelson_siegel_yield(6.0 / 12.0, beta)
            y_12m = get_nelson_siegel_yield(12.0 / 12.0, beta)
            y_24m = get_nelson_siegel_yield(24.0 / 12.0, beta)
            
            # Formatta la stringa in sintassi LaTeX, i "\\" finali chiudono la riga in LaTeX
            print(f"        {beta[0]:.3f} & {beta[1]:.3f} & {beta[2]:.3f} & " +
                  f"{y1m:.4f} & {y3m:.4f} & {y6m:.4f} & {y_12m:.4f} & {y_24m:.4f} \\\\")

if __name__ == "__main__":
    __main__()