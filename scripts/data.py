import pandas as pd
import matplotlib.pyplot as plt

def get_density(plot=True):
    df = pd.read_csv("./gen1/df_gen1.csv")
    df_clean = df[(df["pixel_art"]==False) & (df["color_art"]==False) & (df["column_art"]==False)]
    df_pixel = df[df["pixel_art"]==True]
    df_color = df[df["color_art"]==True]
    df_column = df[df["column_art"]==True]

    len_clean = len(df_clean)
    len_artif = len(df)-len_clean

    len_pixel = len(df_pixel)
    len_color = len(df_color)
    len_column = len(df_column)

    print(f"Artifact distribution is {len_artif*100/len(df):.2f}%")

    print(
        f"Pixel / Color / Column distribution on artifact data is "
        f"{len_pixel*100/len_artif:.2f}% / "
        f"{len_color*100/len_artif:.2f}% / "
        f"{len_column*100/len_artif:.2f}%"
    )

    if plot :
        fig1, ax1 = plt.subplots()
        ax1.bar(
            ["Clean", "Artifact"],
            [len(df_clean), len(df) - len(df_clean)]
            )
        ax1.set_title("Clean and Artifact Distribution")

        fig2, ax2 = plt.subplots()
        ax2.bar(
            ["Pixel artifact", "Color artifact", "Column artifact"],
        [len(df_pixel), len(df_color), len(df_column)]
        )
        ax2.set_title("Artifact Type Distribution")
        
        plt.show()

if __name__ == '__main__':
    get_density()

