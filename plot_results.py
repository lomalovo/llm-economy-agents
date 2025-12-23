import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_latest():
    # Ищем последний созданный csv файл
    list_of_files = glob.glob('data/results/*.csv') 
    if not list_of_files:
        print("Нет данных для построения.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Построение графиков по файлу: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Настройка графиков
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Результаты симуляции: {latest_file}')
    
    # 1. Цены
    axes[0, 0].plot(df['step'], df['avg_price'], marker='o', color='red')
    axes[0, 0].set_title('Средняя Цена (Price Level)')
    axes[0, 0].grid(True)
    
    # 2. Продажи (ВВП)
    axes[0, 1].plot(df['step'], df['total_sales'], marker='o', color='green')
    axes[0, 1].set_title('Продажи (Real GDP Proxy)')
    axes[0, 1].grid(True)
    
    # 3. Безработица
    axes[1, 0].plot(df['step'], df['unemployment_rate'], marker='x', color='orange')
    axes[1, 0].set_title('Безработица')
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].grid(True)
    
    # 4. Денежная масса (проверка багов)
    axes[1, 1].plot(df['step'], df['total_money_supply'], color='blue')
    axes[1, 1].set_title('Денежная масса (Money Supply)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_latest()
