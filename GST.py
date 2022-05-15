import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

st.set_page_config(page_title="GST酶活数据自动处理", layout="wide", page_icon="📈")

# Layout
st.title("📈GST酶活数据自动处理")
TEMPLATE = st.empty()
DATA = st.container()
PROOF = st.sidebar.container()
RESULT = st.container()
TEXT = st.columns(2)


@st.cache
def load_data() -> pd.DataFrame:
    return pd.read_csv(st.session_state.DATA).dropna()


@st.cache
def save_data(name: str, ID: str, Group: str, cls: str, data: pd.DataFrame) -> None:
    data.set_index(data.columns[0]).to_csv(f"./data/{cls}-{Group}-{name}-{ID}.csv")
    return


def check_input() -> None:
    if len(st.session_state.NAME) == 0:
        st.warning("请输入姓名")
        st.stop()
    elif len(st.session_state.ID) < 10:
        st.warning("请输入正确的学号")
        st.stop()
    elif st.session_state.GROUP < 1 or st.session_state.GROUP > 6:
        st.warning("请输入正确的组号")
        st.stop()
    elif st.session_state.DATA is None:
        st.warning("请上传吸光度数据")
        st.stop()
    elif st.session_state.DATA is not None:
        if len(st.session_state.DATA.getvalue()) < 1000:
            st.success("文件校验通过")
        else:
            st.error("文件校验失败，请检查是否正确保存为csv格式")
            st.stop()


# Modeling enzyme kinetics
def model(t, K, S=None):
    if S is None:
        S = st.session_state.S
    epsilon = st.session_state.epsilon
    L = st.session_state.L
    return (1 - np.exp(-K * t)) * S * epsilon * L


def fit_data(Abs):
    t = st.session_state.T
    if st.session_state.fix_total:
        guess = [0.1]
        bounds = ([0], [10])
    else:
        guess = [0.1, st.session_state.S]
        bounds = ([0, 0], [10, st.session_state.S])
    popt, pcov = curve_fit(
        f=model,
        xdata=t,
        ydata=Abs,
        p0=guess,
        bounds=bounds,
    )
    return popt


def fit_and_plot():
    t = st.session_state.T
    Abs = st.session_state.Abs
    popt = fit_data(Abs)
    K_estimate = popt[0]
    if st.session_state.fix_total:
        S_estimate = st.session_state.S
    else:
        S_estimate = popt[1]
    fit = model(t, *popt)
    upper = model(t, *popt * 1.025)
    lower = model(t, *popt * 0.975)
    R_squared = np.corrcoef(Abs, fit)[0, 1] ** 2
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, Abs, "o", color="tab:blue", label="Raw data")
    ax.plot(t, fit, "-", color="tab:orange", label="Fitted curve")
    ax.fill_between(t, lower, upper, color="tab:orange", alpha=0.2)
    ax.set_xlabel("Time(min)")
    ax.set_ylabel("$Abs_{340}$(a.u.)")
    ax.set_title(f"Fit of Abs data, R$^2$={R_squared:f}")
    ax.legend()
    st.pyplot(fig)
    st.write("#### 拟合结果")
    st.write(
        r"""
        $$
        \begin{cases}
        K_{tot}
        """
        f"={K_estimate:f}"
        r"\\S_0"
        f"={S_estimate:f}"
        r"""
        \end{cases}
        $$
        """
    )
    st.write("---")
    return K_estimate, S_estimate


# with TEXT[0]:
with PROOF:
    with st.expander("数学推导", expanded=False):
        st.write("---")
        st.write(
            r"""
            以下会用到的符号的含义如下
            $$
            \begin{cases}
                E &= &Enzyme\\
                S &= &Substrate\\
                P &= &Product\\
            \end{cases}
            $$
            下标带 $_0$ 字样的表示总量或起始值

            被方括号 $[\ ]$ 所包围的表示瞬时浓度

            ---

            我们假设GST所催化的GSH取代反应仅一步中间态，即反应中化学平衡应遵守:
            $$
            E+S \underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}} ES \underset{k_{-2}}{\overset{k_2}{\rightleftharpoons}} E+P
            $$
            为简化条件，不妨假设从中间态到产物的过程几乎不可逆，即有 $k_{-2}\to0$，那么就可以简化为:
            $$
            E+S \underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}} ES \overset{k_2}{\rightharpoonup} E+P
            $$
            由于时间尺度较大，反应体系应当已经达到平衡，我们又不妨假设 $[ES]$ 的大小恒定，即消耗与生成速率相当，则有:
            $$
            (k_{-1}+k_2)[ES] = k_1[E][S]
            $$
            与常规米氏方程推导不同的是，由于GST的催化速率较高，我们不妨假设动态平衡时仅有少量[ES]的存在，因此我们可取近似:
            $$
            [E] = E_0-[ES]\approx E_0
            $$
            其中 $E_0$ 指酶的总浓度，应为一常数。结合上述条件，我们不难得到:
            $$
            \begin{aligned}
                & (k_{-1}+k_2)[ES] = k_1E_0[S]\\
                \Rightarrow& [ES] = \frac{k_1E_0[S]}{k_{-1}+k_2}\\
                \Rightarrow& \frac{d[P]}{dt} = k_2[ES] = \frac{k_1k_2E_0[S]}{k_{-1}+k_2}\\
            \end{aligned}
            $$
            由于我们测得的 $Abs_{340}$ 值可视为只与产物 $[P]$ 有关，即:
            $$
            Abs_{340}=[P]\cdot\varepsilon\cdot L\propto[P]
            $$
            其中 $\varepsilon$ 为消光系数，$L$ 为比色杯光程，因此我们可以通过求得 $[P]$ 随时间的表达式来对 $Abs_{340}$ 进行建模。为此，我们假定加入的底物总浓度为:
            $$
            S_0=[S]+[P]
            $$
            则有:
            $$
            \begin{aligned}
                & \frac{d[P]}{dt} = \frac{k_1k_2E_0(S_0-[P]))}{k_{-1}+k_2}\\
                \Rightarrow& \frac{k_{-1}+k_2}{k_1k_2E_0}\cdot\frac{d[P]}{S_0-[P]} = dt\\
                \xRightarrow{两边积分}& \frac{k_{-1}+k_2}{k_1k_2E_0}\ln{\frac{1}{1-\frac{[P]}{S_0}}} = t\\
                \Rightarrow& \frac{1}{1-\frac{[P]}{S_0}} = \exp\left\{\frac{k_1k_2E_0}{k_{-1}+k_2}\cdot t\right\}\\
                \Rightarrow& 1-\frac{[P]}{S_0} = \exp\left\{-\frac{k_1k_2E_0}{k_{-1}+k_2}\cdot t\right\}\\
                \Rightarrow& [P] = \left(1-\exp\left\{-\frac{k_1k_2E_0}{k_{-1}+k_2}\cdot t\right\}\right)S_0\\
            \end{aligned}
            $$
            我们将 $\frac{k_1 k_2}{k_{-1}+k_2} E_0$ 设为 $K_{tot}$，代表该溶液中酶的总活力，简化后的表达式应为:
            $$
            [P] = \left(1-e^{-K_{tot}\cdot t}\right)S_0\\
            $$
            代入 $Abs_{340}=[P]\cdot\varepsilon\cdot L$ 则有:
            $$
            Abs_{340}=\left(1-e^{-K_{tot}\cdot t}\right)S_0\cdot\varepsilon\cdot L
            $$
            其中 $\varepsilon=9.6 L\cdot mmol^{-1}\cdot cm^{-1}$，$L=1 cm$，$S_0$ 可根据加入GSH的量推算（由于体系中其他因素的影响，这一项也可作为待定项），因此待定系数仅有 $K_{tot}$ 一项，我们可以据此对数据进行模型拟合来求得所有待定项
            """
        )

with TEMPLATE:
    with open("./template.csv", "r") as template:
        st.download_button(
            label="点击下载模板",
            data=template,
            file_name="吸光度数据.csv",
            mime="text/csv",
            help="表中各列分别代表'时间|电动匀浆器|玻璃匀浆器|珠磨均质器|pH6.0|pH6.5|pH7.0|pH7.5'，数据如有复用也请完整填写，0min处数值应为0或近似于0",
        )

with DATA:
    with st.form("基本信息"):
        colName, colId, colClass, colGroup = st.columns(4)
        with colName:
            st.text_input(
                label="姓名",
                key="NAME",
            )
        with colId:
            st.text_input(
                label="学号",
                value="2000000000",
                max_chars=10,
                key="ID",
            )
        with colClass:
            st.selectbox(
                label="班级",
                options=["周四", "周五"],
                key="CLASS",
            )
        with colGroup:
            st.selectbox(
                label="组号",
                options=[1, 2, 3, 4, 5, 6],
                key="GROUP",
            )
        st.file_uploader(
            label="上传吸光度数据",
            type="csv",
            key="DATA",
            accept_multiple_files=False,
            help="上传前请检查文件正确保存为csv格式，正确保存时大小不超过1kb",
        )
        submit = st.form_submit_button(
            label="提交数据",
            help="新提交数据会自动覆盖，请避免重复提交",
        )
        if submit:
            check_input()
            save_data(
                st.session_state.NAME,
                st.session_state.ID,
                st.session_state.GROUP,
                st.session_state.CLASS,
                load_data(),
            )
        else:
            check_input()

    with st.expander("数据选择", expanded=False):
        st.write("---")
        data = load_data()
        st.selectbox(label="选择处理数据", options=data.columns[1:], key="Choice")
        st.session_state.T = data.iloc[:, 0].values
        st.session_state.Abs = data[st.session_state.Choice].values
        st.write("体系参数")
        colsA = [st.columns(3) for _ in range(2)]
        with colsA[0][0]:
            st.number_input(
                label="GSH终浓度(mM)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.001,
                format="%.3f",
                key="S",
            )
        with colsA[0][1]:
            st.number_input(
                label="含酶样品加样量(μL)",
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.001,
                format="%.3f",
                key="E",
            )
        with colsA[0][2]:
            st.number_input(
                label="体系总体积(mL)",
                min_value=0.0,
                max_value=5.0,
                value=3.0,
                step=0.001,
                format="%.3f",
                key="V",
            )
        with colsA[1][0]:
            st.checkbox(
                label="固定GSH浓度",
                value=False,
                help="固定后所输入的GSH终浓度将仅作参考，拟合结果中的浓度可能相差较大",
                key="fix_total",
            )
        with colsA[1][1]:
            st.number_input(
                label="消光系数(ε)",
                min_value=0.0,
                max_value=10.0,
                value=9.6,
                step=0.001,
                format="%.3f",
                key="epsilon",
            )
        with colsA[1][2]:
            st.number_input(
                label="比色杯光程(cm)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.001,
                format="%.3f",
                key="L",
            )
        if st.button(label="按同样参数处理所有数据"):
            df = pd.DataFrame(
                {
                    "Method": [],
                    "Enzyme_Activity_tot": [],
                    "Enzyme_Activity_avg": [],
                    "S_0": [],
                    "R^2": [],
                }
            )
            for col_name in data.columns[1:]:
                popt = fit_data(data[col_name].values)
                K_estimate = popt[0]
                if st.session_state.fix_total:
                    S_estimate = st.session_state.S
                else:
                    S_estimate = popt[1]
                K_tot = (1 - np.exp(-K_estimate)) * S_estimate * st.session_state.V
                K_avg = (
                    (1 - np.exp(-K_estimate))
                    * S_estimate
                    * st.session_state.V
                    / st.session_state.E
                )
                fit = model(st.session_state.T, *popt)
                R_squared = np.corrcoef(data[col_name].values, fit)[0, 1] ** 2
                df = df.append(
                    pd.DataFrame(
                        {
                            "Method": [col_name],
                            "Enzyme_Activity_tot": [K_tot],
                            "Enzyme_Activity_avg": [K_avg],
                            "S_0": [S_estimate],
                            "R^2": [R_squared],
                        }
                    )
                )
            df = df.set_index("Method")
            st.dataframe(df)
            st.download_button(
                label="下载结果",
                data=df.to_csv(),
                file_name="result.csv",
                mime="text/csv",
            )


# with TEXT[1]:
with RESULT:
    with st.expander("计算结果", expanded=False):
        st.write("---")
        K_estimate, S_estimate = fit_and_plot()
        st.write(
            r"""
            求得系数项后，我们不难从中得到所求的平均总酶活，即:
            $$
            \text{总酶活}(\mu mol\cdot min^{-1})=\Delta[P]_{60s}\cdot v
            $$
            其中 $v=3mL$ 为反应体积，根据以上数值计算得到:
            $$
            总酶活=
            """
            f"{(1 - np.exp(-K_estimate)) * S_estimate * st.session_state.V:f}"
            r"""
            \ \mu mol\cdot min^{-1}
            $$
            为了消除不同样品加样量不同所造成的影响，我们还可以将总酶活折算为单位体积酶活，即:
            $$
            \text{单位体积酶活}(\mu mol\cdot min^{-1}\cdot\mu L^{-1})=\Delta[P]_{60s}\cdot\frac{v}{v_s}
            $$
            $v_s$ 为体系中所加样品体积，根据所填数据计算得:
            $$
            \text{单位体积酶活}=
            """
            f"{(1 - np.exp(-K_estimate)) * S_estimate * st.session_state.V / st.session_state.E:f}"
            r"""
            \ \mu mol\cdot min^{-1}\cdot\mu L^{-1}
            $$
            后续假设检验中我们所使用的酶活数据均应使用单位体积酶活。
            """
        )
