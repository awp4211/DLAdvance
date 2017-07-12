# Exception Maximization
## Gaussian Mixture Model
```math
\Theta_{MLE}={\arg{\max}}_\Theta(\sum_{i=1}^n\log\sum_{l=1}^n\alpha_lN(X|\mu_l,\Sigma_l))
```
## Convex Function
```math
f((1-t)x_1+tx_2) \le (1-t)f(x_1)+tf(x_2) \qquad t \in(0 \cdots 1)
```

## Jensens Inequality
函数 `$\Phi$`  
```math
\Phi((1-t)x_1+tx_2) \leq (1-t)\Phi(x_1)+t\Phi(x_2) \qquad t \in (0 \cdots 1)

\Longrightarrow \Phi(p_1x_1+p_2x_2+...+p_nx_x) \leq p_1\Phi(x_1)+p_2\Phi(x_2)+...+p_n\Phi(x_n) 

\Longrightarrow \Phi(\sum_{i=1}^np_if(x_i))  \leq \sum_{i=1}^np_i\Phi(f(x_i))

s.t. \qquad \sum_{i=1}^np_i=1
```
如果函数连续且 `$\int_{x \in S}p(x)=1$`,有:
```math
\Phi(\int_{x \in S}f(x)p(x)) \leq \int_{x \in S}\Phi(f(x_i))p(x)

\Longrightarrow \Phi \mathbb{E}[f(x)]< \mathbb{E}\Phi(f(x_i))]
```
For example:`$\Phi(x)=-\log(x)$`is a convex fucntion,
`$-\log \mathbb{E}(f(x)) \leq \mathbb{E}[-\log f(x))]$`

## EM算法
最大似然估计中：参数`$\theta$`使用下面的公式进行估计：
```math
\theta^{MLE}=\arg\max_{\theta}(\mathcal{L}(\theta))=\arg\max_{\theta}(\log[p(\mathbf{X}|\theta)])
```
- EM算法添加隐变量latent variable 'Z'到模型中
- 参数`$\Theta=\{\theta^1,\theta^2,\ldots,\theta^n\}$`

每次循环中更新EM算法中参数的规则为：
```math
\Theta^{(g+1)}=\arg\max_\theta\bigg(\int_{\mathbf{Z}} \log(p(\mathbf{X,Z}|\theta))p(\mathbf{Z}|\mathbf{X}, \Theta^{(g)}) \mathrm{d}\mathbf{Z} \bigg)
```

- EM算法的收敛性
```math
\begin{aligned}
    \mathcal{L}(\theta|\mathbf{X})&=\ln(p(\mathbf{X}|\theta)) \\
    &=\ln(\frac{p(\mathbf{X,Z}|\theta)}{p(\mathbf{Z}|\mathbf{X},\theta)}) \\
    &=\ln\bigg(\frac{\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})}}
              {\frac{p(\mathbf{Z}|\mathbf{X},\theta)}{Q(\mathbf{Z})}}\bigg)\\
    &=\ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})} \times   
                \frac{Q(\mathbf{Z})}{p(\mathbf{Z}|\mathbf{X},\theta)}\bigg)\\
    &=\ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})}        \bigg)
     +\ln\bigg(\frac{Q(\mathbf{Z})}{p(\mathbf{Z}|\mathbf{X},\theta)} \bigg) \\
   \Longrightarrow \ln(p(\mathbf{X}|\theta)) &= 
        \int_{\mathbf{Z}}     \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})} \bigg)Q(\mathbf{Z}) \mathrm{d}\mathbf{Z}
        +
        \int_{\mathbf{Z}}\ln\bigg(\frac{Q(\mathbf{Z})}{p(\mathbf{Z}|\mathbf{X},\theta)} \bigg)Q(\mathbf{Z})\mathrm{d}\mathbf{Z} \\
    &=\int_{\mathbf{Z}}                 \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})} \bigg)Q(\mathbf{Z}) \mathrm{d}\mathbf{Z}
        +
        \mathbb{KL}[Q(\mathbf{Z}) \| p(\mathbf{Z}|\mathbf{X},\theta)] \\
    \stackrel{\mathbb{KL}>0}{\Longrightarrow} 
        \mathcal{L}(\theta|\mathbf{X}) &= \ln(p(\mathbf{X}|\theta)) 
            \ge  \int_{\mathbf{Z}}
                 \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})} \bigg)Q(\mathbf{Z}) \mathrm{d}\mathbf{Z} \\
    \stackrel{Jensen Inequality}{\Longrightarrow} 
        \mathcal{L}(\theta|\mathbf{X}) &= \ln(p(\mathbf{X}|\theta)) 
        =\ln \int_\mathbf{Z}p(\mathbf{X,Z}|\theta) \\
    &= \ln \int_\mathbf{Z}          \frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})}Q(\mathbf{Z}) \\
    &\ge \int_\mathbf{Z} \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})} \bigg)Q(\mathbf{Z}) 
\end{aligned}
```
E-M算法变成M-M算法：
```math
\begin{aligned}
   \ln(p(\mathbf{X}|\theta)) &= 
        \int_{\mathbf{Z}}     \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})} \bigg)Q(\mathbf{Z}) \mathrm{d}\mathbf{Z}
        +
        \int_{\mathbf{Z}}\ln\bigg(\frac{Q(\mathbf{Z})}{p(\mathbf{Z}|\mathbf{X},\theta)} \bigg)Q(\mathbf{Z})\mathrm{d}\mathbf{Z} \\
    &=\mathbf{F}(\Theta,Q) + \mathbb{KL}[Q(\mathbf{Z}) \| p(\mathbf{Z}|\mathbf{X},\theta)]
\end{aligned}
```
1. 固定`$\Theta=\Theta^{(g)}$`,最大化`$Q(\mathbf{Z})$` :
-  `$\mathcal{L}(\Theta|\mathbf{X})$`是`$\mathbf{F}(\Theta,Q)$`的上界
-  为了使`$\mathcal{L}(\Theta|\mathbf{X})=\mathbf{F}(\Theta,Q)$`，`$\mathbb{KL}[Q(\mathbf{Z}) \| p(\mathbf{Z}|\mathbf{X},\theta)]=0$`。所以选择`$Q(\mathbf{Z})=p(\mathbf{Z}|\mathbf{X},\theta)$`，此时

```math
\begin{aligned}
    \mathcal{L}(\Theta|\mathbf{X})&=\int_{\mathbf{Z}}     \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{Q(\mathbf{Z})} \bigg)Q(\mathbf{Z}) \mathrm{d}\mathbf{Z} \\
    &=\int_{\mathbf{Z}}              \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{p(\mathbf{Z}|\mathbf{X},\theta)} \bigg)p(\mathbf{Z}|\mathbf{X},\theta) \mathrm{d}\mathbf{Z} \\
    &=\int_{\mathbf{Z}}              \ln\bigg(\frac{p(\mathbf{X,Z}|\theta)}{p(\mathbf{Z}|\mathbf{X},\theta^{(g)})} \bigg)p(\mathbf{Z}|\mathbf{X},\theta^{(g)}) \mathrm{d}\mathbf{Z}
\end{aligned}
```
2. 固定`$Q(\mathbf{Z})$`最大化`$\Theta$`
```math
\Theta^{(g+1)}
=\arg\max_\theta\bigg(\int_{\mathbf{Z}} \log(p(\mathbf{X,Z}|\theta))p(\mathbf{Z}|\mathbf{X}, \Theta^{(g)}) \mathrm{d}\mathbf{Z} \bigg)
=\arg\max_\theta\bigg(Q(\Theta,\Theta^{(g)}) \bigg)
```


所以`$\color{red}{\mathcal{L(\Theta^{(g+1)})} \ge \mathcal{L(\Theta^{(g)})}}$`,

## GMM in EM
Gaussian Mixture Model
```math
p(\mathbf{X}|\Theta)
=\sum_{l=1}^k\alpha_l\mathcal{N}(\mathbf{X}|\mu_l,\Sigma_l)
=\prod_{i=1}^n \sum_{l=1}^k\alpha_l\mathcal{N}(x_i|\mu_l,\Sigma_l)

p(\mathbf{X,Z}|\Theta)
=\prod_{i=1}^n p(x_i,z_i|\Theta)
=\prod_{i=1}^n p(x_i|z_i,\Theta) p(z_i|\Theta)
=\prod_{i=1}^n \mathcal{N}(\mu_{z_i},\Sigma_{z_i}) \times \alpha_{z_i}


p(\mathbf{Z}|\mathbf{X},\Theta)
=\prod_{i=1}^n p(z_i|x_i, \Theta)
=\prod_{i=1}^n \frac{\alpha_{z_i} \mathcal{N}(\mu_{z_i},\Sigma_{z_i})}
            {\sum_{l=1}^k \alpha_{l} \mathcal{N}(\mu_{l},\Sigma_{l})}
```
- One Theorem
```math
\text{Joint PDF} \quad P(\mathbf{Y})
=P(y_1,\ldots,y_n)
=\prod_{i=1}^nP(y_i)

F(\mathbf{Y})
=f_1(y_1)+\ldots+f_n(y_n)
=\sum_{i=1}^n f_i(y_i)

\int_\mathbf{Y}(F(\mathbf{Y}))P(\mathbf{Y})\mathrm{d}\mathbf{Y}
=\int_{y_1}\int_{y_2} \ldots \int_{y_n} 
    \bigg( \sum_{i=1}^n (f_i(y_i)) \bigg) P(\mathbf{Y}) 
    \mathrm{d}y_1 \ldots \mathrm{d}y_1

\begin{aligned}
    \text{First Term} &=\int_{y_1}\int_{y_2} \ldots \int_{y_n} 
        \bigg( f_1(y_1) \bigg) P(\mathbf{Y}) \prod_{i=1}^n \mathrm{d}y_i \\
    &= \int_{y_1}f_1(y_1) \bigg(\int_{y_2} \ldots \int_{y_n}P(y_1,y_2,     \ldots y_n) \prod_{i=2}^n (\mathrm{d}y_i) \bigg)\mathrm{y}_1 \\
    &= \int_{y_1}f_1(y_1)p(y_1)\mathrm{d}y_1 \\
    \Longrightarrow \int_\mathbf{Y}(F(\mathbf{Y}))P(\mathbf{Y})\mathrm{d}\mathbf{Y} &= 
        \sum_{i=1}^n(\int_{y_i}f_i(y_i)p_i(y_i)\mathrm{d}y_i)
\end{aligned}
```


- E-Step:
```math
\begin{aligned}
Q(\Theta,\Theta^{(g)}) &= \int_{\mathbf{Z}}         \log(p(\mathbf{X,Z}|\theta))p(\mathbf{Z}|\mathbf{X}, \Theta^{(g)}) \mathrm{d}\mathbf{Z} \\
&= \int_{z_1} \ldots \int_{z_n}\bigg(\sum_{i=1}^n \ln p(x_i|z_i,\Theta) \prod_{i=1}^n p(z_i|x_i, \Theta^{(g)}) \bigg) \mathrm{d}z_1 \ldots \mathrm{d}z_n \\
&= \sum_{i=1}^n \bigg(\int_{z_i}\ln p(x_i|z_i,\Theta) p(z_i|x_i, \Theta^{(g)})\mathrm{d}z_i \bigg) \qquad z_i \in (1 \ldots k) \\
&= \sum_{i=1}^n \sum_{z_i=1}^k \ln p(x_i|z_i,\Theta) p(z_i|x_i, \Theta^{(g)}) \\
&= \sum_{z_i=1}^k \sum_{i=1}^n  \ln p(x_i|z_i,\Theta) p(z_i|x_i, \Theta^{(g)}) \\
&= \sum_{l=1}^k \sum_{i=1}^n \ln[\alpha_l\mathcal{N}(x_i|\mu_l,\Sigma_l)]p(l|x_i,\Theta^{(g)}) \\
&= \sum_{l=1}^k \sum_{i=1}^n \ln(\alpha_l)p(l|x_i,\Theta^{(g)})
+\sum_{l=1}^k \sum_{i=1}^n \ln(\mathcal{N}(x_i|\mu_l,\Sigma_l))p(l|x_i,\Theta^{(g)})
\end{aligned}
```
- Linear Algorithm
```math
\sum_i x_i^TAx_i=tr(A\sum_i x_ix_i^T)

\frac{\partial\ln|X|}{\partial X}=2X^{-1}-diag(X^{-1})

\text{when X is symmetric,}\frac{\partial tr(XB)}{\partial X}=B+B^T-diag(B)

\text{In general} \frac{\partial tr(XB)}{\partial X}=B
```

- M-Step:
```math
\begin{aligned}
Q(\Theta,\Theta^{(g)})^{Term1}
&=\sum_{l=1}^k \sum_{i=1}^n \ln(\alpha_l)p(l|x_i,\Theta^{(g)})
\text{  related to } \alpha \\
\mathbb{LM}(\alpha_1,\ldots,\alpha_n, \lambda)
&=\sum_{l=1}^k \sum_{i=1}^n \ln(\alpha_l)p(l|x_i,\Theta^{(g)})
-\lambda(\sum_{l=1}^k \alpha_l-1) \\
&=\sum_{l=1}^k \ln(\alpha_l) \bigg(\sum_{i=1}^n p(l|x_i,\Theta^{(g)}) \bigg)-\lambda(\sum_{l=1}^k \alpha_l-1) \\
\Longrightarrow \frac{\partial \mathbb{LM}}{\partial \alpha_l}
&= \frac{1}{\alpha_l}\bigg(\sum_{i=1}^n p(l|x_i,\Theta^{(g)}) \bigg)-\lambda=0 \\
\alpha_l &= \frac{1}{N}\bigg(\sum_{i=1}^n p(l|x_i,\Theta^{(g)}) \bigg)
\end{aligned}
```

```math
\begin{aligned}
Q(\Theta,\Theta^{(g)})^{Term2}
&=\sum_{l=1}^k \sum_{i=1}^n \ln(\mathcal{N}(x_i|\mu_l,\Sigma_l))p(l|x_i,\Theta^{(g)})
\text{  related to } \mu \enspace \Sigma \\
\frac{\partial Q(\Theta,\Theta^{(g)})^{Term2}}{\partial \mu_1,\ldots,\partial\mu_k,\partial \Sigma_1,\ldots,\partial \Sigma_k}
&=[0,\ldots,0] \\
Q(\Theta,\Theta^{(g)})^{Term2}&=\sum_{l=1}^kS(\mu_l,\Sigma_l^{-1}) \\
S(\mu_l,\Sigma_l^{-1})
&=\sum_{i=1}^n\ln(\mathcal{N}(x_i|\mu_l,\Sigma_l))p(l|x_i,\Theta^{(g)}) \\
&=\sum_{i=1}^n \bigg( -\frac{1}{2} \ln(|\Sigma_l|)-\frac{1}{2}(x_i-\mu_l)^T\Sigma^{-1}(x_i-\mu_l) \bigg)
p(l|x_i,\Theta^{(g)}) \\
& \Longrightarrow -Tr\bigg(\frac{\Sigma_l^{-1}}{2}\sum_{i=1}^n (x_i-\mu_l) (x_i-\mu_l)^Tp(l|x_i,\Theta^{(g)} \bigg)+\text{Contstant} \\
\text{1.}\frac{\partial S(\mu_l,\Sigma_l^{-1})}{\partial{\mu_l}}
&=\frac{2\Sigma_l^{-1}-diag(\Sigma^{-1})}{2}\sum_{i=1}^n2(x_i-\mu_l)p(l|x_i,\Theta^{(g)})=0 \\
& \Longrightarrow \sum_{i=1}^nx_ip(l|x_i,\Theta^{(g)})
=\mu_l \sum_{i=1}^np(l|x_i,\Theta^{(g)}) \\
& \Longrightarrow \mu_l=\frac{\sum_{i=1}^n p(l|x_i,\Theta^{(g)}) }{\sum_{i=1}^n x_ip(l|x_i,\Theta^{(g)})} \\
\text{2.}S(\mu_l,\Sigma_l^{-1})
&=\frac{1}{2}\sum_{i=1}^n \ln(|\Sigma_l^{-1}|)p(l|x_i,\Theta^{(g)})
-\frac{1}{2}Tr\bigg(\Sigma^{-1}\sum_{i=1}^n(x_i-\mu_l)(x_i-\mu_l)^Tp(l|x_i,\Theta^{(g)}) \bigg) \\
\Longrightarrow \frac{\partial S(\mu_l,\Sigma_l^{-1})}{\partial \Sigma_l^{-1}} 
&=\frac{2\sum_{i=1}^n \Sigma_l p(l|x_i,\Theta^{(g)})-\sum_{i=1}^ndiag(\Sigma_l)p(l|x_i,\Theta^{(g)})}{2} \\
&- \frac{2\sum_{i=1}^n(x_i-\mu_l)(x_i-\mu_l)^Tp(l|x_i,\Theta^{(g)})
    - diag(sum_{i=1}^n(x_i-\mu_l)(x_i-\mu_l)^Tp(l|x_i,\Theta^{(g)}))
}{2}=0 \\
\Longrightarrow \Sigma_l &= \frac{\sum_{i=1}^n(x_i-\mu_l)(x_i-\mu_l)^Tp(l|x_i,\Theta^{(g)})}
{\sum_{i=1}^np(l|x_i,\Theta^{(g)})}
\end{aligned}
```
