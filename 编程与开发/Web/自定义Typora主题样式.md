# 自定义Typora主题样式

在Windows平台上，Tpyora软件的主题样式文件存储在\$HOME/AppData/Roaming/Typora/themes目录下，一个自定义样式bln-dark.css文件内容如下。

```css
/* Bloonow Dark Typora Theme. */
/*
<html>
<head></head>
<body>
    <header>
        <div id="top-titlebar" class="stopselect">标题栏</div>
        <div class="megamenu-menu stopselect" id="megamenu-menu-sidebar">标题菜单栏或侧边菜单栏</div>
    </header>
    <content>
        <div id="write" class="enable-diagrams">编辑区域</div>
        <div id="md-searchpanel" class="form-inline searchpanel-replace-mode">搜索替换面板</div>
        <ul class="dropdown-menu context-menu show" id="context-menu-for-source">源码模式右键菜单</ul>
        <ul class="dropdown-menu context-menu show" id="context-menu">右键菜单</ul>
        <ul class="dropdown-menu context-menu ext-context-menu show" id="insert-menu">右键次级菜单</ul>
    </content>
    <footer class="stopselect ty-footer">状态栏</footer>
    <div id="megamenu-content" class="megamenu-content stopselect">
        菜单面板
        <div class="megamenu-section" id="megamenu-section-open"></div>
        <div class="megamenu-section hide" id="megamenu-section-export"></div>
    </div>
</body>
</html>
*/

/* 自定义颜色变量 */
:root {
    --theme-bg-color: #424554;
    --theme-text-color: #C0EBBD;
    --theme-color: #8F499A;
    --theme-color-alpha: rgba(197, 112, 216, 0.4);
    --theme-side-color: #353846;
    --dark-color: #09020C;
    --gray-color: #ADADAD;
    --light-color: #FEFEFE;
    --hint-text-color: #54DB4F;
    --head-text-color: #86D060;
    --link-text-color: #C16124;
    --refer-text-color: #C9CBD2;
    --del-text-color: #b2b8a3;
    --code-text-color: #A9B7C6;
    --code-bg-color: #19292D;
    --latex-bg-color: #1f202A;
    --mark-bg-color: #F0F000;
    --table-bg-color: #406864;
    --select-bg-color: #7A88A3;
    --search-bg-color: #333333;
}

/* Typora预定义的颜色变量名称 */
:root {
    /* 背景色 */
    --bg-color: var(--theme-bg-color);
    /* 文本颜色 */
    --text-color: var(--theme-text-color);
    /* 专注模式下其它段落蒙版文本颜色 */
    --blur-text-color: var(--gray-color);
    /* 诸如LaTeX公式源码块区域背景色 */
    --rawblock-edit-panel-bd: var(--latex-bg-color);
    /* 选择区域背景色 */
    --select-text-bg-color: var(--select-bg-color);
    /* 选择区域文本颜色 */
    --select-text-font-color: auto;
    /* 搜索选择区域背景色 */
    --search-select-bg-color: var(--search-bg-color);
    /* 搜索选择区域文本颜色 */
    --search-select-text-color: var(--hint-text-color);
    /* 元数据内容颜色 */
    --meta-content-color: var(--theme-color);
    /* 主题色 */
    --primary-color: var(--theme-color);
    /* 主题色之按钮边界 */
    --primary-btn-border-color: var(--theme-color);
    /* 主题色之按钮文本颜色 */
    --primary-btn-text-color: var(--theme-color);
    /* 侧边菜单栏背景色 */
    --side-bar-bg-color: var(--theme-side-color);
    /* 控制文本颜色 */
    --control-text-color: var(--light-color);
    /* 打开文件的背景色 */
    --active-file-bg-color: var(--theme-side-color);
    /* 打开文件的文本颜色 */
    --active-file-text-color: var(--hint-text-color);
    /* 打开文件的边界颜色 */
    --active-file-border-color: var(--theme-color);
    /* 条目悬浮背景色 */
    --item-hover-bg-color: var(--theme-color);
    /* 条目悬浮文本颜色 */
    --item-hover-text-color: inherit;
}

@font-face {
    font-family: "ChillJinshuSong";
    src: url("C:/Users/bln/AppData/Local/Microsoft/Windows/Fonts/ChillJinshuSong.otf");
}

@media print {
    html {
        font-size: 10px;
    }

    pre,
    table {
        /* 避免插入分页符 */
        page-break-inside: avoid;
        /* 在长单词或URL地址内部进行换行 */
        word-wrap: break-word;
    }
}

html {
    font-size: 16px;
}

body {
    font-family: "ChillJinshuSong", "Consolas", "Times New Roman", "SimSun";
    line-height: 1.6;
}

/* 编辑区域 */
#write {
    max-width: 1200px;
    margin: 0 auto;
    padding: 30px;
    padding-bottom: 300px;
}

/* 元数据，即用于控制样式的符号，渲染时不会显示，例如 # * - $ ` [ ] \ 等 */
#write .md-meta {
    color: var(--hint-text-color);
}

/* 目录区域 */
#write .md-toc {
    margin: 20px 0;
    color: var(--theme-color);
}

/* 设置元素通用边界 */
#write p,
#write blockquote,
#write ul,
#write ol,
#write dl,
#write table {
    margin: 0.5rem 0;
}

/* 标题 */
#write h1,
#write h2,
#write h3,
#write h4,
#write h5,
#write h6 {
    display: block;
    font-weight: bold;
    color: var(--head-text-color);
}

#write h1 {
    color: #5CC428;
    font-size: 2.4rem;
    margin: 1rem 0;
    line-height: 1.2;
    text-align: center;
    border-bottom: 1px solid var(--light-color);
}

#write h2 {
    font-size: 2.0rem;
    margin: 0.4rem 0;
}

#write h3 {
    font-size: 1.6rem;
    margin: 0.5rem 0;
}

#write h4 {
    font-size: 1.3rem;
    margin: 0.6rem 0;
}

#write h5 {
    font-size: 1.2rem;
    margin: 0.7rem 0;
}

#write h6 {
    font-size: 1.1rem;
    margin: 0.95rem 0;
}

/* 水平横线 */
#write hr {
    height: 2px;
    padding: 0;
    margin: 0.75rem 0;
    background-color: var(--theme-color);
    border: 0 none;
    overflow: hidden;
    box-sizing: content-box;
}

/* 超链接 */
#write a {
    color: var(--link-text-color);
    text-decoration: none;
}

/* 超链接，鼠标悬停 */
#write a:hover {
    text-decoration: underline;
}

/* 引用块区域 */
#write blockquote {
    border-left: 2px solid var(--theme-color);
    font-size: 1rem;
    padding-left: 1rem;
    color: var(--refer-text-color);
}

/* 高亮文本区域 */
#write mark {
    background-color: var(--mark-bg-color);
    color: var(--dark-color);
    border-radius: 2px;
}

/* 删除文本区域 */
#write del,
#write s {
    color: var(--del-text-color);
}

/* 脚注 */
#write .md-footnote {
    background-color: var(--code-bg-color);
    color: var(--gray-color);
}

/* 代码块区域 */
#write .md-fences {
    background-color: var(--code-bg-color);
    color: var(--code-text-color);
    font-size: 0.95rem;
    margin: 0.2rem 0;
    padding: 0.25rem 0;
    border-radius: 3px;
    border: none;
    text-shadow: none;
}

#write .md-fences .code-tooltip {
    background-color: var(--code-bg-color);
}

/* 内联代码 */
#write code,
#write tt {
    background-color: var(--code-bg-color);
    color: var(--code-text-color);
    font-size: 1rem;
    margin: 0 2px;
    border-radius: 2px;
    border: 1px solid var(--theme-color-alpha);
}

/* 表格 */
#write table {
    padding: 0;
    width: 100%;
    border-collapse: collapse;
    word-break: initial;
}

#write thead {
    background-color: var(--table-bg-color);
}

#write table th {
    text-align: center;
    padding: 0.25rem 0.25rem;
    border: 1px solid var(--theme-color);
}

#write table td {
    padding: 0.25rem 0.25rem;
    border: 1px solid var(--theme-color);
}

/* 列表 */
#write ul:first-child,
#write ol:first-child {
    margin-top: 0;
}

#write ul:last-child,
#write ol:last-child {
    margin-bottom: 0;
}

#write ul,
#write ol {
    margin-left: 0;
    padding-left: 1.25rem;
}

#write ul>li::marker,
#write ol>li::marker {
    color: var(--theme-color);
}

/* 状态栏 */
footer.ty-footer {
    border: none;
}

/* 首选项区域 */
.ty-preferences .window-header {
    background-color: var(--active-file-bg-color);
}

.ty-preferences .nav-group-item:hover {
    background-color: var(--item-hover-bg-color);
}

/* 右侧滚动条 */
::-webkit-scrollbar-thumb {
    border-radius: 5px;
    background-color: var(--control-text-color);
}

/* 侧边菜单栏 */
.megamenu-menu {
    background-color: var(--side-bar-bg-color);
    box-shadow: none;
}

/* 菜单面板 */
.megamenu-content,
.megamenu-opened header {
    background: var(--bg-color);
    background-image: none;
}

.megamenu-menu-panel table td:nth-child(1) {
    background-color: var(--bg-color);
}

.megamenu-menu-panel table td:nth-child(2) {
    background-color: var(--bg-color);
}

/* 代码高亮样式 */
.cm-s-inner.CodeMirror {
    background-color: var(--code-block-bg-color);
}

.cm-s-inner .CodeMirror-gutters {
    background: var(--code-block-bg-color);
    color: #464B5D;
    border: none;
}

.cm-s-inner .CodeMirror-linenumber {
    color: #464B5D;
}

.cm-s-inner .CodeMirror-guttermarker {
    color: #FFEE80;
}

.cm-s-inner .CodeMirror-guttermarker-subtle {
    color: #D0D0D0;
}


.cm-s-inner .CodeMirror-cursor {
    border-left: 1px solid #A9B7C6;
}

.cm-s-inner div.CodeMirror-cursor {
    border-left: 1px solid #ffffff;
}

.cm-s-inner div.CodeMirror-selected {
    background: rgba(113, 124, 180, 0.2);
}

.cm-s-inner.CodeMirror-focused div.CodeMirror-selected {
    background: rgba(113, 124, 180, 0.2);
}

.cm-s-inner .CodeMirror-selected {
    background: #214283 !important;
}

.cm-s-inner .CodeMirror-selectedtext {
    background: #214283 !important;
}

.cm-overlay.CodeMirror-selectedtext {
    background: #B5D6FC !important;
}

.cm-s-inner .CodeMirror-line::selection,
.cm-s-inner .CodeMirror-line>span::selection,
.cm-s-inner .CodeMirror-line>span>span::selection {
    background: rgba(128, 203, 196, 0.2);
}

.cm-s-inner .CodeMirror-line::-moz-selection,
.cm-s-inner .CodeMirror-line>span::-moz-selection,
.cm-s-inner .CodeMirror-line>span>span::-moz-selection {
    background: rgba(128, 203, 196, 0.2);
}

.cm-s-inner .cm-keyword {
    color: #C792EA;
}

.cm-s-inner .cm-operator {
    color: #89DDFF;
}

.cm-s-inner .cm-variable-2 {
    color: #EEFFFF;
}

.cm-s-inner .cm-variable-3,
.cm-s-inner .cm-type {
    color: #f07178;
}

.cm-s-inner .cm-builtin {
    color: #FFCB6B;
}

.cm-s-inner .cm-atom {
    color: #F78C6C;
}

.cm-s-inner .cm-number {
    color: #FF5370;
}

.cm-s-inner .cm-def {
    color: #82AAFF;
}

.cm-s-inner .cm-string {
    color: #caf18f;
}

.cm-s-inner .cm-string-2 {
    color: #f07178;
}

.cm-s-inner .cm-comment {
    color: #578958;
}

.cm-s-inner .cm-variable {
    color: #f07178;
}

.cm-s-inner .cm-tag {
    color: #FF5370;
}

.cm-s-inner .cm-meta {
    color: #FFCB6B;
}

.cm-s-inner .cm-attribute {
    color: #C792EA;
}

.cm-s-inner .cm-property {
    color: #C792EA;
}

.cm-s-inner .cm-qualifier {
    color: #DECB6B;
}

.cm-s-inner .cm-variable-3,
.cm-s-inner .cm-type {
    color: #DECB6B;
}


.cm-s-inner .cm-error {
    color: rgba(255, 255, 255, 1.0);
    background-color: #FF5370;
}

.cm-s-inner .CodeMirror-matchingbracket {
    text-decoration: underline;
    color: white !important;
}
```

