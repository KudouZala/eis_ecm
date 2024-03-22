#!/usr/bin/env python# 使用env的Python解释器，指定UTF-8编码格式
# coding: utf-8
#--------------------------------------------------------------------------------------------------------------
# ## Set up the environment 🔌# 设置环境
# AutoEIS relies on `EquivalentCircuits.jl` package to perform the EIS analysis.
# The package is not written in Python, so we need to install it first.
# AutoEIS ships with `julia_helpers` module that helps to install and manage Julia dependencies with minimal user interaction. For convenience, installing Julia and the required packages is done automatically when you import `autoeis` for the first time. If you have Julia installed already (discoverable in system PATH), it'll get detected and used, otherwise, it'll be installed automatically.

# AutoEIS依赖于Julia语言编写的`EquivalentCircuits.jl`包来进行电化学阻抗谱(EIS)分析。
# 因为这个包不是用Python写的，所以我们需要先安装它。
# AutoEIS随附了`julia_helpers`模块，它帮助最小化用户交互来安装和管理Julia依赖。
# 为了方便，当你第一次导入`autoeis`时，会自动完成Julia和所需包的安装。
# 如果你已经安装了Julia（在系统PATH中可以找到），则会使用已安装的Julia，否则会自动进行安装。

# In[1]:
from multiprocessing import freeze_support

import os
import sys

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import seaborn as sns
from IPython.display import display

import autoeis as ae

#----------------------------------------------------------------------------------------------------------
# # AutoEIS demo

def dropdown_trace_plots():# 创建下拉菜单和输出窗口用于选择电路并显示其迹图
    """Creates a dropdown menu to select a circuit and plot its trace."""
    def on_dropdown_clicked(change):# 创建下拉菜单和输出窗口用于选择电路并显示其迹图
        with output:
            output.clear_output()# 清除之前的输出
            idx = circuits_str.index(change.new)# 获取选中电路的索引
            plot = trace_plots[idx]# 获取对应的迹图
            display(plot) # 显示迹图
    # 创建下拉菜单
    dropdown = widgets.Dropdown(description='Circuit:', options=circuits_str, value=None)
    output = widgets.Output()
    # 设置下拉菜单的观察者，当选项变化时触发函数
    dropdown.observe(on_dropdown_clicked, names="value")
    display(dropdown, output)
# In[ ]:

# 定义一个交互式下拉菜单，可以选择一个电路模型，并显示其参数的后验分布和迹图。
# 根据下拉菜单选择的模型，显示相关的图表。首次运行这个单元可能需要一些时间（每个电路约5秒#），但一旦运行完毕，所有的图表将被缓存起来。
def plot_trace(samples):# 定义函数，用于绘制MCMC采样器中变量的后验分布和迹图
    """Plots the posterior and trace of a variable in the MCMC sampler."""
    output = widgets.Output() # 创建输出窗口
    with output:
        fig, ax = plt.subplots(ncols=2, figsize=(9, 3))# 创建两列的子图：一列用于后验分布，一列用于迹图
        log_scale = bool(np.std(samples) / np.median(samples) > 2)# 如果标准差与中位数的比值大于2，则使用对数刻度
        kwargs_hist = { # 设置直方图参数
            "stat": "density",
            "log_scale": log_scale,
            "color": "lightblue",
            "bins": 25
        }
        # ax[0] -> posterior, ax[1] -> trace
        # 绘制后验分布直方图
        sns.histplot(samples, **kwargs_hist, ax=ax[0])
        kwargs_kde = {"log_scale": log_scale, "color": "red"}# 绘制核密度估计曲线
        sns.kdeplot(samples, **kwargs_kde, ax=ax[0])
        # Plot trace# 绘制迹图
        ax[1].plot(samples, alpha=0.5)
        ax[1].set_yscale("log" if log_scale else "linear")
        #plt.show()
        plt.close(fig)  # 关闭当前图形
    return output

# 定义函数，绘制所有变量的后验分布和迹图
def plot_trace_all(mcmc: "numpyro.MCMC", circuit: str):
    """Plots the posterior and trace of all variables in the MCMC sampler."""
    # 获取电路模型参数标签
    variables = ae.parser.get_parameter_labels(circuit)
    samples = mcmc.get_samples()# 获取MCMC采样的样本
    children = [plot_trace(samples[var]) for var in variables]# 获取MCMC采样的样本
    tab = widgets.Tab()
    tab.children = children
    tab.titles = variables # 设置标签页标题
    return tab

# In[ ]:

# 定义一个函数，用于绘制使用后验中位数的电路的Nyquist图
def plot_nyquist(mcmc: "numpyro.MCMC", circuit: str,freq,Z):
    """Plots Nyquist plot of the circuit using the median of the posteriors."""
    # Compute circuit impedance using median of posteriors # 用后验分布的中位数计算电路阻抗
    samples = mcmc.get_samples()
    variables = ae.parser.get_parameter_labels(circuit)
    percentiles = [10, 50, 90]
    # 计算参数的百分位数
    params_list = [[np.percentile(samples[v], p) for v in variables] for p in percentiles]
    # 生成电路函数
    circuit_fn = ae.utils.generate_circuit_fn(circuit)
    # 使用百分位数参数计算模拟阻抗
    Zsim_list = [circuit_fn(freq, params) for params in params_list]
    # Plot Nyquist plot
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p, Zsim in zip(percentiles, Zsim_list):# 绘制Nyquist图
        ae.visualization.plot_nyquist(Zsim, fmt="-", label=f"model ({p}%)", ax=ax)
    ae.visualization.plot_nyquist(Z, "o", label="measured", ax=ax)
    # Next line is necessary to avoid plotting the first time
    #plt.close(fig)  # 关闭当前图形
    #plt.show()# 为了避免第一次绘图时就显示，需要关闭图形
    return fig

# 定义一个函数，创建一个下拉菜单，用于选择电路并绘制其Nyquist图
def dropdown_nyquist_plots(circuits_str,nyquist_plots):
    """Creates a dropdown menu to select a circuit and plot its Nyquist plot."""

    # 当选择改变时调用此函数
    def on_change(change):
        with output:
            output.clear_output()# 清除之前的输出
            idx = circuits_str.index(change.new)# 获取选择的电路的索引
            fig = nyquist_plots[idx] # 获取对应的Nyquist图
            display(fig)# 显示Nyquist图

    # 创建输出窗口
    output = widgets.Output()
    # 创建下拉菜单，选项是电路字符串列表
    dropdown = widgets.Dropdown(options=circuits_str, value=None, description='Circuit:')
    # 当下拉选择变化时更新输出
    dropdown.observe(on_change, names='value')
    display(dropdown, output)

def main():
    # 设置绘图风格
    ae.visualization.set_plot_style()

    # Set this to True if you're running the notebook locally
    # 如果你在本地运行notebook，请将这个变量设置为True
    interactive = False

    #--------------------------------------------------------------------------------------------------------------
    # ## Load EIS data 📈

    # Once the environment is set up, we can load the EIS data.
    # You can use [`pyimpspec`](https://vyrjana.github.io/pyimpspec/guide_data.html) to load EIS data from a variety of popular formats.
    # Eventually, AutoEIS requires two arrays: `Z` and `freq`. `Z` is a complex impedance array, and `freq` is a frequency array.
    # Both arrays must be 1D and have the same length.
    # The impedance array must be in Ohms, and the frequency array must be in Hz.
    #
    # Here, we use `numpy` to load the test data from a txt file, which contains the frequency array in the first column and the impedance array in the second and third columns (Re and -Im parts). We then convert the impedance array to complex numbers, and it should be ready to use.
    # ##加载EIS数据
    # 完成环境设置后，我们可以加载EIS数据。
    # 你可以使用[`pyimpspec`](https://vyrjana.github.io/pyimpspec/guide_data.html)从多种流行的格式加载EIS数据。
    # 最终，AutoEIS需要两个数组：`Z`和`freq`。
    # `Z`是一个复阻抗数组，`freq`是一个频率数组。两个数组必须是1D的并且长度相同。
    # 阻抗数组必须以欧姆为单位，频率数组必须以赫兹为单位。

    # 在这里，我们使用`numpy`从txt文件加载测试数据，
    # 文件中第一列是频率数组，第二和第三列是阻抗数组的实部和负虚部。
    # 然后我们将阻抗数组转换为复数，它就可以使用了。
    # In[2]:


    # Load impedance data (skip header row); Columns 1 -> 3: frequency, Re(Z), Im(Z)
    # 加载阻抗数据（跳过标题行）；第1 -> 3列：频率，Re(Z)，Im(Z)
    ASSETS = ae.io.get_assets_path()# 获取资源路径
    fpath = os.path.join(ASSETS, "test_data.txt") # 拼接文件完整路径
    # 加载文件，跳过第一行标题，只取第0、1、2列数据，分别对应频率、阻抗的实部和虚部
    freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=1, unpack=True, usecols=(0, 1, 2))
    # Convert to complex impedance (the file contains -Im(Z) hence the minus sign)
    # 转换为复阻抗（文件中存储的是负虚部，因此需要减号）
    Z = Zreal - 1j*Zimag#Z是一个矩阵，元素都是复数



    #--------------------------------------------------------------------------------------------------------------
    # Now let's plot the data using AutoEIS's built-in plotting function. The function takes the impedance array and the frequency array as inputs. It will plot the impedance spectrum in the Nyquist plot and the Bode plot. All plotting functions in AutoEIS can either be directly called or an `Axes` object can be passed in to specify the plotting location.
    # 现在我们使用AutoEIS内置的绘图函数来绘制数据。
    # 该函数接收阻抗数组和频率数组作为输入。
    # 它将在Nyquist图和Bode图中绘制阻抗谱。
    # AutoEIS中的所有绘图函数都可以直接调用，也可以传入一个`Axes`对象来指定绘图位置。

    # In[3]:

    # 绘制Nyquist图和Bode图
    fig, ax = ae.visualization.plot_impedance_combo(Z, freq)
    #plt.show()
    """
    在 plot_impedance_combo 函数中，图像是通过 matplotlib 创建的。如果你在一个标准的Python脚本中运行此函数，它不会自动显示图像。为了在脚本执行结束时看到图像，你需要调用 plt.show() 来启动事件循环，并且显示所有活跃的图像对象。
    如果你在Jupyter Notebook中运行这个函数，并且你的环境正确设置了 matplotlib 的inline后端，图像应该会自动显示在你运行代码的单元格下方。
    如果你在一个脚本或其他环境中运行此函数，请确保在函数调用之后添加 plt.show()，
    """
    # Alternatively, you can manually create a subplot and pass it to the plotting function
    # Make sure to create two columns for the two plots (Nyquist and Bode)
    #或者，您可以手动创建子图并将其传递给绘图函数
    #确保为两个图(Nyquist和Bode)创建两个列
    #fig, ax = plt.subplots(ncols=2)
    #ae.visualization.plot_impedance_combo(Z, freq, ax=ax)


    #--------------------------------------------------------------------------------------------------------------
    # ## EIS analysis 🪄
    # ## Preprocess impedance data 🧹
    # 在进行EIS分析之前，我们需要对阻抗数据进行预处理。
    # Before performing the EIS analysis, we need to preprocess the impedance data. The preprocessing step is to remove outliers. AutoEIS provides a function to perform the preprocessing. As part of the preprocessing, the impedance measurements with a positive imaginary part are removed, and the rest of the data are filtered using linear KK validation. The function returns the filtered impedance array and the frequency array.
    #预处理步骤是去除异常值。
    #AutoEIS提供了一个函数来执行预处理。
    #在预处理过程中，将移除具有正虚部的阻抗测量值，并使用线性KK验证过滤剩余的数据。
    #该函数返回过滤后的阻抗数组和频率数组。
    # In[4]:


    Z, freq, rmse = ae.core.preprocess_impedance_data(Z, freq, threshold=5e-2, plot=True)


    #--------------------------------------------------------------------------------------------------------------
    # ##生成候选等效电路
    # ## Generate candidate equivalent circuits 📐
    #
    # In this stage, AutoEIS generates a list of candidate equivalent circuits using a customized genetic algorithm (done via the package `EquivalentCircuits.jl`). The function takes the filtered impedance array and the frequency array as inputs. It returns a list of candidate equivalent circuits. The function has a few optional arguments that can be used to control the number of candidate circuits and the circuit types. The default number of candidate circuits is 10, and the default circuit types are resistors, capacitors, constant phase elements, and inductors. The function runs in parallel by default, but you can turn it off by setting `parallel=false`.
    #
    # > Note: Since running the genetic algorithm can be time-consuming, we will use a pre-generated list of candidate circuits in this demo to get you started quickly. If you want to generate the candidate circuits yourself, set `use_pregenerated_circuits=False` in the cell below.
    # 在这个阶段，AutoEIS使用定制的遗传算法（通过EquivalentCircuits.jl包实现）生成一系列候选等效电路。
    # 该函数接受过滤后的阻抗数组和频率数组作为输入，返回一个候选等效电路列表。
    # 该函数有一些可选参数，可以用来控制候选电路的数量和电路类型。
    # 默认的候选电路数量为10，电路类型包括电阻器、电容器、恒相元件和电感器。
    # 函数默认以并行方式运行，但你可以通过设置parallel=false来关闭并行计算。
    # 注意：由于运行遗传算法可能非常耗时，我们将在这个演示中使用预先生成的候选电路列表来快速开始。
    # 如果你想自己生成候选电路，可以在下面的单元中将use_pregenerated_circuits设置为False。
    # In[5]:

    #如果选择使用预先生成的电路，将该变量设置为True
    use_pregenerated_circuits = False
    # 根据变量决定是加载预生成的电路还是现场生成
    if use_pregenerated_circuits:
        circuits_unfiltered = ae.io.load_test_circuits() # 加载预先生成的电路
    else:
        # 设置遗传算法的参数

        kwargs = {
            "iters": 25,# 迭代次数
            "complexity": 25,# 电路复杂度
            "population_size": 100,# 种群大小
            "generations": 50,# 世代数
            "tol": 1e-3,# 容忍度阈值
            "parallel":True# 是否并行计算#这里一定要注意写True，否则训练出的CSV，在后面读取的时候进行贝叶斯推理的时候会报错
        }


        # 生成候选电路
        #circuits_unfiltered = ae.core.generate_equivalent_circuits(Z, freq, **kwargs)
        # Since generating circuits is expensive, let's save the results to a CSV file
        # 由于生成电路代价高昂，我们将结果保存到CSV文件中
        #circuits_unfiltered.to_csv("circuits_unfiltered.csv", index=False)
        # To load from file, uncomment the next 2 lines (line 2 is to convert str -> Python objects)
        # 若要从文件中加载，取消注释下面两行（第二行是将字符串转换成Python对象）
        circuits_unfiltered = pd.read_csv("circuits_unfiltered.csv")
        circuits_unfiltered["Parameters"] = circuits_unfiltered["Parameters"].apply(eval)
    # 显示未经过滤的电路
    #circuits_unfiltered

    #--------------------------------------------------------------------------------------------------------------
    # ## Filter candidate equivalent circuits 🧹
    #
    # Note that all these circuits generated by the GEP process probably fit the data well, but they may not be physically meaningful. Therefore, we need to filter them to find the ones that are most plausible. AutoEIS uses "statistical plausibility" as a proxy for gauging "physical plausibility". To this end, AutoEIS provides a function to filter the candidate circuits based on some heuristics (read our [paper](https://doi.org/10.1149/1945-7111/aceab2) for the exact steps and the supporting rationale).
    # ##过滤候选等效电路
    # 注意，通过GEP过程生成的所有电路可能都很好地拟合数据，但它们可能在物理上没有意义。
    # 因此，我们需要过滤它们以找到最有可能的电路
    # AutoEIS使用“统计可信度”作为衡量“物理合理性”的代理。
    # 为此，AutoEIS提供了一个函数，根据一些启发式规则过滤候选电路
    # （阅读我们的[论文](https://doi.org/10.1149/1945-7111/aceab2)了解具体步骤和支持的理由）。

    # In[7]:


    # 调用AutoEIS核心模块的函数过滤不合理的电路，基于统计学的启发式规则
    #circuits = ae.core.filter_implausible_circuits(circuits_unfiltered)
    # Let's save the filtered circuits to a CSV file as well
    # 将过滤后的电路保存到CSV文件，以便之后使用
    #circuits.to_csv("circuits_filtered.csv", index=False)
    # To load from file, uncomment the next 2 lines (line 2 is to convert str -> Python objects)
    # 若要从文件加载过滤后的电路，取消注释下面两行（第二行是将字符串转换成Python对象）
    circuits = pd.read_csv("circuits_filtered.csv")
    circuits["Parameters"] = circuits["Parameters"].apply(eval)
    #circuits# 显示过滤后的电路列表


    #--------------------------------------------------------------------------------------------------------------
    # ## Perform Bayesian inference 🧮
    # # 执行贝叶斯推断
    # Now that we have narrowed down the candidate circuits to a few good ones, we can perform Bayesian inference to find the ones that are statistically most plausible.
    # 现在我们已经缩小了候选电路的范围，我们可以执行贝叶斯推断，找出统计上最可能的电路。
    # In[8]:

    # 调用AutoEIS核心模块的函数进行贝叶斯推断
    mcmc_results = ae.core.perform_bayesian_inference(circuits, freq, Z)
    mcmcs, status = zip(*mcmc_results)# 解压结果，获取MCMC对象和状态

    #--------------------------------------------------------------------------------------------------------------
    # ## Visualize results 📊
    #
    # Now, let's take a look at the results. `perform_bayesian_inference` returns a list of `MCMC` objects. Each `MCMC` object contains all the information about the Bayesian inference, including the posterior distribution, the prior distribution, the likelihood function, the trace, and the summary statistics.
    #
    # Before we visualize the results, let's take a look at the summary statistics. The summary statistics are the mean, the standard deviation, and the 95% credible interval of the posterior distribution. The summary statistics are useful for quickly gauging the uncertainty of the parameters.
    # 可视化结果
    # 我们来看一下结果。`perform_bayesian_inference`返回一个包含所有贝叶斯推断信息的`MCMC`对象列表，
    # 包括后验分布、先验分布、似然函数、迹和摘要统计。
    # In[9]:

    # 遍历MCMC对象列表，打印每个电路模型的摘要统计信息
    for mcmc, stat, circuit in zip(mcmcs, status, circuits.circuitstring):
        if stat == 0:
            ae.visualization.print_summary_statistics(mcmc, circuit)


    # Note that some rows have been highlighted in yellow, indicating that the standard deviation is greater than the mean. This is not necessarily a bad thing, but it screams "caution" due to the high uncertainty. In this case, we need to check the data and the model to see if there is anything wrong. For example, the data may contain outliers, or the model may be overparameterized.
    #
    # Now, let's take one step further and visualize the results. To get an overview of the results, we can plot the posterior distributions of the parameters as well as the trace plots. It's an oversimplification, but basically, a good posterior distribution should be unimodal and symmetric, and the trace plot should be stationary. In probabilistic terms, this means that given the circuit model, the data are informative about the parameters, and the MCMC algorithm has converged.
    #
    # On the other hand, if the posterior distribution is multimodal or skewed, or the trace plot is not stationary, it means that the data are not informative about the parameters, and the MCMC algorithm has not converged. In this case, we need to check the data and the model to see if there is anything wrong. For example, the data may contain outliers, or the model may be overparameterized.
    #
    # > For the following cell to work, you need to set `interactive=True` at the beginning of the notebook. It's turned off by default since GitHub doesn't render interactive plots.
    # 注意一些行被高亮显示为黄色，指示标准偏差大于均值。这不一定是坏事，但它表明不确定性很高。
    # 在这种情况下，我们需要检查数据和模型，看看是否有什么问题。例如，数据可能包含异常值，或者模型可能过度参数化。

    # 现在，让我们进一步可视化结果。为了快速查看结果，我们可以绘制参数的后验分布和迹图。
    # 这是一种简化，但基本上，一个好的后验分布应该是单峰的和对称的，迹图应该是平稳的。
    # 在概率术语中，这意味着给定电路模型，数据对参数是有信息的，MCMC算法已经收敛。

    # 另一方面，如果后验分布是多峰的或偏斜的，或者迹图不是平稳的，
    # 这意味着数据对参数没有信息，MCMC算法没有收敛。在这种情况下，我们需要检查数据和模型，
    # 看看是否有什么问题。例如，数据可能包含异常值，或者模型可能过度参数化。

    # > 要运行以下单元格，你需要在notebook开头设置`interactive=True`。
    # 这是因为GitHub不渲染交互式图表，默认关闭。



    # 缓存渲染的图表以避免重复渲染
    # Cache rendered plots to avoid re-rendering
    circuits_str = circuits["circuitstring"].tolist()
    trace_plots = []
    for mcmc, stat, circuit in zip(mcmcs, status, circuits_str):
        if stat == 0:
            trace_plots.append(plot_trace_all(mcmc, circuit))
        else:
            trace_plots.append("Inference failed")
    # 如果设置为交互式，显示下拉菜单和迹图
    #if interactive:
        #dropdown_trace_plots()


    # The functions defined in the above cell are used to make the interactive dropdown menu. The dropdown menu lets you select a circuit model, and shows the posterior distributions of the parameters as well as the trace plots. The dropdown menu is useful for quickly comparing the results of different circuit models. Running this cell for the first time may take a while (~ 5 seconds per circuit), but once run, all the plots will be cached.
    #
    # The distributions for the most part look okay, although in some cases (like R2 and R4 in the first circuit) the span is quite large (~ few orders of magnitude). Nevertheless, the distributions are bell-shaped. The trace plots also look stationary.
    #
    # Now, let's take a look at the posterior predictive distributions. "Posterior predictive" is a fancy term for "model prediction", meaning that after we have performed Bayesian inference, we can use the posterior distribution to make predictions. In this case, we can use the posterior distribution to predict the impedance spectrum and compare it with our measurements and see how well they match. After all, all the posteriors might look good (bell-shaped, no multimodality, etc.) but if the model predictions don't match the data, then the model is not good.
    #
    # > For the following cell to work, you need to set `interactive=True` at the beginning of the notebook. It's turned off by default since GitHub doesn't render interactive plots.
    # 以上单元格定义了用于创建交互式下拉菜单的函数。下拉菜单允许您选择电路模型，
    # 并显示参数的后验分布和迹图。下拉菜单便于快速比较不同电路模型的结果。
    # 分布大部分看起来是可以的，虽然在某些情况下（如第一个电路中的R2和R4）跨度很大（约几个数量级）。
    # 尽管如此，分布是钟形的。迹图也看起来是平稳的。

    # 现在，让我们来看看后验预测分布。"后验预测"是"模型预测"的一个高级


    # Cache rendered plots to avoid re-rendering# 缓存Nyquist图以避免重复渲染
    circuits_str = circuits["circuitstring"].tolist()
    nyquist_plots = []
    a=0
    for mcmc, stat, circuit in zip(mcmcs, status, circuits_str):
        a=a+1
        if stat == 0:
            nyquist_plots.append(plot_nyquist(mcmc, circuit,freq,Z))
        else:
            nyquist_plots.append("Inference failed")
    # 如果设置为交互式，显示下拉菜单和Nyquist图
    plt.show()
    print(a)

if __name__ == "__main__":
    freeze_support()
    main()