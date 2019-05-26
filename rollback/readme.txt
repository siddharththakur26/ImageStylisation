\subsubsection{Execution}

The program was developed under and are compatible with:
\begin{itemize}
\item Python 3.6
\item PyTorch 1.0.1.post2
\item CentOS 7 x86\_64
\end{itemize}
To run the code, execute shell commands like
\begin{verbatim}
python36 0.py 5000
\end{verbatim}
where 5000 is the number of desired training cycles to reach.
The different hyperparameters will be automatically covered.
The program automatically picks up stored models.
To start fresh, clear the stored models and outputs by
\begin{verbatim}
rm out/*/*.*
\end{verbatim}
but do not remove the directories.


