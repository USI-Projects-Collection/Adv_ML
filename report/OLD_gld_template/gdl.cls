% \NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gdl}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass{article}

\RequirePackage{times}
\usepackage[scaled=.95]{helvet}

\RequirePackage{graphicx, hyperref}
\usepackage[square,sort,comma,numbers]{natbib}

\RequirePackage{latexsym,ifthen,rotating,calc,textcase,booktabs,color,endnotes}
\RequirePackage{amsfonts,amssymb,amsbsy,amsmath,amsthm}
\RequirePackage[errorshow]{tracefnt}

\flushbottom
\frenchspacing

\usepackage[text={174mm,258mm},%
papersize={210mm,297mm},%
columnsep=12pt,%
headsep=21pt,%
centering]{geometry}
\usepackage{ftnright}

\parindent 1em

\hyphenpenalty=1000
\pretolerance=8000
\tolerance=9500
\hbadness=8000
\vbadness=9000
\displaywidowpenalty=0
\clubpenalty=10000
\widowpenalty=10000
\lefthyphenmin=3%
\righthyphenmin=3%
\brokenpenalty=10000%

\thinmuskip = 3mu
\medmuskip = 4mu
\thickmuskip = 5mu

\setcounter{topnumber}{10}
\def\topfraction{1}
\setcounter{bottomnumber}{10}
\def\bottomfraction{0.8}
\setcounter{totalnumber}{10}
\def\textfraction{0}
\renewcommand{\floatpagefraction}{0.95}
\setcounter{dbltopnumber}{10}
\renewcommand{\dblfloatpagefraction}{0.95}
\renewcommand{\dbltopfraction}{1}

\renewcommand\small{%
   \@setfontsize\small\@ixpt{10.5}%
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               %\parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand{\refsize}{\fontsize{9}{12pt}\selectfont}
\renewcommand{\footnotesize}{\fontsize{8}{10pt}\selectfont}
\renewcommand{\scriptsize}{\fontsize{7.0}{8.5pt}\selectfont}
\renewcommand{\large}{\fontsize{12}{14pt}\selectfont}
\newcommand{\titlesize}{\fontsize{18.0}{20pt}\selectfont}

\newbox\absbox
\def\abstract{\lrbox\absbox\minipage{\textwidth}%
  \normalsize%
  \section*{\normalsize Abstract}\vskip -1.5mm%
  }
\def\endabstract{\endminipage\endlrbox}

\def\keywords#1{%
  \gdef\@keywords{\begin{minipage}{\textwidth}{\normalsize \textbf{Keywords}}\\ \parbox[t]{\textwidth}{#1}\end{minipage}}}
\let\@keywords\@empty

\skip\footins 20pt plus  8pt
%\gdef\footnoterule{}
\def\footnoterule{\kern-3\p@
  \hrule \@width \columnwidth \kern 3mm}

\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\long\def\@makefntext#1{\parindent 1em%
\noindent{$\m@th^{\@thefnmark}$}#1}

\newcommand{\email}[1]{%
  \gdef\@email{%
\footnotetext[0]{ Email: #1}}}
\let\@email\@empty

\def\corrauth#1{\gdef\@corrauth{%
\footnotetext[0]{\par\vskip-3pt\noindent\textbf{Corresponding author:}\\ #1}}}
\let\@corrauth\@empty

\def\affiliation#1{%
  \gdef\@affiliation{%
    \footnotetext[0]{ #1}}}
\let\@affiliation\@empty

\def\affilnum#1{${}^{\text{{#1}}}$}

\renewcommand\maketitle{\par
  \begingroup
     \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
       \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{title}\label{FirstPage}\@affiliation\@corrauth\@email%
  \endgroup
  %\setcounter{footnote}{0}%
    \global\let\affiliation\relax
   \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\@affiliation\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
\vspace*{-34pt}%
\null%
\begin{center}
\begin{sf}
\begin{minipage}[t]{\textwidth-5cm}
  \vskip 12.5pt%
    {\raggedright\titlesize\textbf{\@title} \par}%
    \vskip 1.5em%
    \vskip 12.5mm%
    \end{minipage}\hfill
    \begin{minipage}[t]{4.5cm}
  \vskip 12.5pt%
    {\hfill\raggedright\textbf{GDL 2025}\\[0.1cm]%
    \hfill\small Group id:~~\groupid\par%
    \hfill\small Project id:~~\projectid}%
    \end{minipage}
{\par%
      \lineskip .5em%
      {\raggedright\@author\par}}
     \vskip 40pt%
    {\noindent\usebox\absbox\par}
    {\vspace{20pt}%
      %
      {\noindent\normalsize\@keywords}\par}
      \end{sf}
      \end{center}
      \vspace{22pt}
        \par%
  }
 
\def\startpage{\pageref{FirstPage}}
\def\endpage{\pageref{LastPage}}

\def\runninghead{\markboth{GDL 2025}}

\def\ps@title{%
\def\@oddhead{\parbox{\textwidth}{\mbox{}\\[-1pt]%
\noindent\rule{\textwidth}{0.5pt}%
}}%
\let\@evenhead\@oddhead
\def\@oddfoot{\parbox[t]{\textwidth}{~}}
\let\@evenfoot\@oddfoot}

\def\ps@sagepage{%
\let\@mkboth\@gobbletwo
\def\@evenhead{\parbox{\textwidth}{%
\normalsize\thepage\hfill\itshape Group id:~\groupid~~--~~Project id:~\projectid \\[-6pt]
\noindent\rule{\textwidth}{0.25pt}}}
\def\@oddhead{\parbox{\textwidth}{%
\sf\leftmark\hfill\thepage\\[-6pt]
\noindent\rule{\textwidth}{0.25pt}}}
\def\@evenfoot{\parbox[t]{\textwidth}{\sf Group id:~\groupid~~--~~Project id:~\projectid}}
\def\@oddfoot{\@evenfoot}
}

%\renewcommand{\@seccntformat}[1]{{\csname the#1\endcsname.}\hspace{0.5em}}
\setcounter{secnumdepth}{2}

\newdimen\@bls                              
\@bls=\baselineskip

\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {1.2\@bls  plus .3\@bls minus .1\@bls}%
                                   {5pt\@afterindentfalse}%
                                   {\sf\large\bfseries\raggedright}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {0.9\@bls plus .3\@bls minus .1\@bls}%
                                     {4pt\@afterindentfalse}%
                                     {\sf\bfseries\raggedright}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {0.5\@bls plus .3\@bls minus .1\@bls}%
                                     {-0.5em\@afterindentfalse}%
                                     {\sf\normalsize\itshape}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{1em}%
                                     {0.3\@bls plus .3\@bls minus .1\@bls}%
                                     {-0.5em\@afterindentfalse}%
                                     {\normalsize\itshape}}

\def\enumerate{\ifnum \@enumdepth >3 \@toodeep\else
      \advance\@enumdepth \@ne
      \edef\@enumctr{enum\romannumeral\the\@enumdepth}\list
      {\csname label\@enumctr\endcsname}{\usecounter
        {\@enumctr}\itemsep 0pt\parsep 0pt
         \def\makelabel##1{\hss\llap{##1}}}\fi}

\let\endenumerate =\endlist

\def\itemize{\ifnum \@itemdepth >3 \@toodeep\else \advance\@itemdepth \@ne
\edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
\list{\csname\@itemitem\endcsname}{\itemsep 0pt\parsep 0pt
\def\makelabel##1{\hss\llap{##1}}}\fi}

\let\enditemize =\endlist

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{sageperiod}{.\hspace*{1ex}}
\captionsetup[figure]{font=small,labelfont={sf,bf},textfont=sf,labelsep=sageperiod,justification=raggedright,singlelinecheck=false}
\captionsetup[table]{position=top,labelsep=sageperiod,font=small,labelfont={sf,bf},textfont=sf,justification=raggedright,skip=2pt,singlelinecheck=false}

\def\@begintheorem#1#2[#3]{%
  \deferred@thm@head{\the\thm@headfont \thm@indent
    \@ifempty{#1}{\let\thmname\@gobble}{\let\thmname\@iden}%
    \@ifempty{#2}{\let\thmnumber\@gobble}{\let\thmnumber\@iden}%
    \@ifempty{#3}{\let\thmnote\@gobble}{\let\thmnote\@iden}%
    \thm@swap\swappedhead\thmhead{#1}{#2\the\thm@headpunct}{#3}%
    %\the\thm@headpunct
    \thmheadnl % possibly a newline.
    \hskip\thm@headsep
  }%
  \ignorespaces}

\def\thmhead@plain#1#2#3{%
  \thmname{#1}\thmnumber{\@ifnotempty{#1}{ }{#2}}%
  \thmnote{ {\the\thm@notefont #3. }}}
  
%Endnotes
\def\enotesize{\refsize}

\def\enoteheading{\subsection*{\normalsize\bfseries\notesname}%
  \mbox{}\par\vskip-\baselineskip}

%\renewcommand\makeenmark{\theenmark.\hspace{0.75em}}

\renewcommand{\enoteformat}{%
   \rightskip\z@ \leftskip15pt \parindent=0pt
   \leavevmode{\makebox[0cm][r]{%
   \hbox to15pt{\@theenmark.\hfill}}}}


\AtEndDocument{%
\label{LastPage}}

\pagestyle{sagepage}

\normalsize
\sloppy
