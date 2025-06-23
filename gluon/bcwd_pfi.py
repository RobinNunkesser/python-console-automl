import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Alle Dateien im Verzeichnis ../data suchen, die 'pfi' im Dateinamen enthalten
pfi_files = glob.glob(os.path.join('../data', '*PFI*'))

for file_path in pfi_files:
    print(f"Verarbeite Datei: {file_path}")
    # with open(file_path, 'r') as f:
    #     content = f.read()
    #     content = 'feature,importance\n' + content
    #     #content = content.replace('0,', '0.')
    #     with open(file_path, 'w') as f:
    #         f.write(content)
    df = pd.read_csv(file_path)
    #print(df.head())
    #print("Enthält Spaltennamen:")
    df['importance'] = df['importance'] * -1
    df_sorted = df.sort_values('importance', ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_sorted, y='feature', x='importance', orient='h')
    plt.title('Feature Importance')
    plt.tight_layout()
    all = True
    for col in ['radius3', 'concave_points3', 'concavity1', 'texture3']:
        if (not (col in df['feature'].values)):
            all = False
    if all:
        plt.savefig(file_path.replace('.csv', '.pdf'))
#    plt.show()
