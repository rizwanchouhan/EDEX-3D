

import pandas as pd
import matplotlib.pyplot as plt

# Define the data from the LaTeX table
data = {
    "Method": ["VOCA", "MeshTalk", "FaceFormer", "FaceTalk", "Ours"],
    "CASIA_LSV": [4.485, 2.785, 2.753, 2.563, 2.244],
    "CASIA_TCE": [4.534, 3.323, 2.755, 2.675, 2.345],
    "CASIA_EDE": [4.232, 2.943, 2.443, 2.434, 2.412],
    "CMU-MOSEI_LSV": [4.213, 3.532, 2.654, 2.543, 2.433],
    "CMU-MOSEI_TCE": [3.865, 3.677, 2.433, 2.467, 2.454],
    "CMU-MOSEI_EDE": [4.185, 2.675, 2.442, 2.436, 2.422],
    "BIWI_LSV": [4.232, 3.675, 2.754, 2.697, 2.323],
    "BIWI_TCE": [3.343, 2.753, 2.423, 2.421, 2.235],
    "BIWI_EDE": [4.829, 3.323, 2.786, 2.694, 2.232],
    "RAVDESS_LSV": [3.574, 3.135, 2.753, 2.686, 2.245],
    "RAVDESS_TCE": [3.497, 2.643, 2.864, 2.784, 2.188],
    "RAVDESS_EDE": [4.321, 3.432, 2.543, 2.534, 2.434],
    "VOCA_LSV": [3.586, 2.885, 2.673, 2.657, 2.124],
    "VOCA_TCE": [4.205, 3.432, 2.724, 2.643, 2.332],
    "VOCA_EDE": [3.781, 3.312, 2.896, 2.754, 2.318]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Plot each dataset separately
datasets = ["CASIA", "CMU-MOSEI", "BIWI", "RAVDESS", "VOCA"]
methods = ["LSV", "TCE", "EDE"]

for dataset in datasets:
    fig, axs = plt.subplots(figsize=(8, 6))
    df_plot = df[["Method", f"{dataset}_LSV", f"{dataset}_TCE", f"{dataset}_EDE"]].set_index("Method")
    df_plot.plot(kind="bar", ax=axs, color=['#ff7f0e', '#17becf', '#9467bd'], width=0.7)
    axs.set_title(f"The experimental analysis based on LSV, TCE, and EDE using the {dataset} Dataset")
    axs.set_ylabel("Values")
    axs.set_xlabel("Method")
    axs.grid(False)
    plt.xticks(rotation=45)
    plt.legend(title="Method")
    # Annotate each bar with its specific value
    for p in axs.patches:
        axs.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Get handles and labels for the legend
    handles, labels = axs.get_legend_handles_labels()
    # Keep only the error names in the legend labels
    legend_labels = ['LSV', 'TCE', 'EDE']
    # Set the legend with custom labels
    axs.legend(handles, legend_labels)
    plt.tight_layout()
    plt.savefig(f"{dataset}_comparison.svg", format="svg")  # Save as SVG
    plt.show()