import seaborn as sns
import matplotlib.pyplot as plt

def summarize_data(df):
    """
    Prints basic dataset structure, summary statistics, and missing values.
    """
    print("\n===== DATA INFO =====")
    print(df.info())

    print("\n===== SUMMARY STATISTICS =====")
    print(df.describe(include='all'))

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())


def plot_histograms(df, columns, save_dir=None):
    """
    Creates histograms for a list of numerical columns.
    """
    for col in columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()

        if save_dir:
            path = f"{save_dir}/{col}_hist.png"
            plt.savefig(path, dpi=150)
            print(f"[SAVED] {path}")

        plt.close()


def plot_box(df, x, y, save_dir=None):
    """
    Creates boxplots for category vs numeric variable.
    """
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(f"{x} vs {y}")
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/{x}_vs_{y}.png"
        plt.savefig(path, dpi=150)
        print(f"[SAVED] {path}")

    plt.close()


def plot_scatter(df, x, y, save_dir=None):
    """
    Creates a scatterplot with a regression line.
    """
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x=x, y=y)
    sns.regplot(data=df, x=x, y=y, scatter=False, color='red')
    plt.title(f"{x} vs {y}")
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/{x}_vs_{y}.png"
        plt.savefig(path, dpi=150)
        print(f"[SAVED] {path}")

    plt.close()
