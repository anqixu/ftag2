




<!DOCTYPE html>
<html>
  <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# object: http://ogp.me/ns/object# article: http://ogp.me/ns/article# profile: http://ogp.me/ns/profile#">
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>ftag2/CMakeLists.txt at master · anqixu/ftag2</title>
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub" />
    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub" />
    <link rel="apple-touch-icon" sizes="57x57" href="/apple-touch-icon-114.png" />
    <link rel="apple-touch-icon" sizes="114x114" href="/apple-touch-icon-114.png" />
    <link rel="apple-touch-icon" sizes="72x72" href="/apple-touch-icon-144.png" />
    <link rel="apple-touch-icon" sizes="144x144" href="/apple-touch-icon-144.png" />
    <meta property="fb:app_id" content="1401488693436528"/>

      <meta content="@github" name="twitter:site" /><meta content="summary" name="twitter:card" /><meta content="anqixu/ftag2" name="twitter:title" /><meta content="ftag2 - Fourier Tag v2: Spacial-Temporally Graceful Degradable Fiducial Markers" name="twitter:description" /><meta content="https://1.gravatar.com/avatar/1c58f846277eb3c24514d1704ce75f39?d=https%3A%2F%2Fidenticons.github.com%2F75b39591eb63db064e81cb175865214f.png&amp;r=x&amp;s=400" name="twitter:image:src" />
<meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="https://1.gravatar.com/avatar/1c58f846277eb3c24514d1704ce75f39?d=https%3A%2F%2Fidenticons.github.com%2F75b39591eb63db064e81cb175865214f.png&amp;r=x&amp;s=400" property="og:image" /><meta content="anqixu/ftag2" property="og:title" /><meta content="https://github.com/anqixu/ftag2" property="og:url" /><meta content="ftag2 - Fourier Tag v2: Spacial-Temporally Graceful Degradable Fiducial Markers" property="og:description" />

    <meta name="hostname" content="github-fe140-cp1-prd.iad.github.net">
    <meta name="ruby" content="ruby 2.1.0p0-github-tcmalloc (87c9373a41) [x86_64-linux]">
    <link rel="assets" href="https://github.global.ssl.fastly.net/">
    <link rel="conduit-xhr" href="https://ghconduit.com:25035/">
    <link rel="xhr-socket" href="/_sockets" />


    <meta name="msapplication-TileImage" content="/windows-tile.png" />
    <meta name="msapplication-TileColor" content="#ffffff" />
    <meta name="selected-link" value="repo_source" data-pjax-transient />
    <meta content="collector.githubapp.com" name="octolytics-host" /><meta content="collector-cdn.github.com" name="octolytics-script-host" /><meta content="github" name="octolytics-app-id" /><meta content="4A0CCF06:3CFA:43E639:53166E03" name="octolytics-dimension-request_id" /><meta content="6013159" name="octolytics-actor-id" /><meta content="dacocp" name="octolytics-actor-login" /><meta content="8533d79e92ebcfc4dcd0738a390f6f74029e67a5aaa318e7dfb75f5bda4d1be1" name="octolytics-actor-hash" />
    

    
    
    <link rel="icon" type="image/x-icon" href="/favicon.ico" />

    <meta content="authenticity_token" name="csrf-param" />
<meta content="yzEaw0uwT2crelLCGD7HTJ5I6A/Uxr1ixBFeGkzqFJE=" name="csrf-token" />

    <link href="https://github.global.ssl.fastly.net/assets/github-144f6fbd93cc11df8cc03bc000351f75fa1e5875.css" media="all" rel="stylesheet" type="text/css" />
    <link href="https://github.global.ssl.fastly.net/assets/github2-e78754b61713d6654d18e641a08d640c76bd5ba2.css" media="all" rel="stylesheet" type="text/css" />
    
    


      <script crossorigin="anonymous" src="https://github.global.ssl.fastly.net/assets/frameworks-490867a148cf94468c141093ccd4e56ed2144eb9.js" type="text/javascript"></script>
      <script async="async" crossorigin="anonymous" src="https://github.global.ssl.fastly.net/assets/github-3e40bad6e91fbbace9ebaa606f72e6fb100546c0.js" type="text/javascript"></script>
      
      <meta http-equiv="x-pjax-version" content="a101843396cb8eb0fa5af14c202201ae">

        <link data-pjax-transient rel='permalink' href='/anqixu/ftag2/blob/6e89d7b603e8956d7a3538ea4e4d2b1a0014108a/CMakeLists.txt'>

  <meta name="description" content="ftag2 - Fourier Tag v2: Spacial-Temporally Graceful Degradable Fiducial Markers" />

  <meta content="1885004" name="octolytics-dimension-user_id" /><meta content="anqixu" name="octolytics-dimension-user_login" /><meta content="13760217" name="octolytics-dimension-repository_id" /><meta content="anqixu/ftag2" name="octolytics-dimension-repository_nwo" /><meta content="false" name="octolytics-dimension-repository_public" /><meta content="false" name="octolytics-dimension-repository_is_fork" /><meta content="13760217" name="octolytics-dimension-repository_network_root_id" /><meta content="anqixu/ftag2" name="octolytics-dimension-repository_network_root_nwo" />
  <link href="https://github.com/anqixu/ftag2/commits/master.atom?token=6013159__eyJzY29wZSI6IkF0b206L2FucWl4dS9mdGFnMi9jb21taXRzL21hc3Rlci5hdG9tIiwiZXhwaXJlcyI6Mjk3MTkwMjA4M30=--bfcd19378c4ea5e9381541ef6567e669c8a64770" rel="alternate" title="Recent Commits to ftag2:master" type="application/atom+xml" />

  </head>


  <body class="logged_in  env-production linux vis-private page-blob tipsy-tooltips">
    <div class="wrapper">
      
      
      
      


      <div class="header header-logged-in true">
  <div class="container clearfix">

    <a class="header-logo-invertocat" href="https://github.com/">
  <span class="mega-octicon octicon-mark-github"></span>
</a>

    
    <a href="/notifications" aria-label="You have no unread notifications" class="notification-indicator tooltipped tooltipped-s" data-gotokey="n">
        <span class="mail-status all-read"></span>
</a>

      <div class="command-bar js-command-bar  in-repository">
          <form accept-charset="UTF-8" action="/search" class="command-bar-form" id="top_search_form" method="get">

<input type="text" data-hotkey="/ s" name="q" id="js-command-bar-field" placeholder="Search or type a command" tabindex="1" autocapitalize="off"
    
    data-username="dacocp"
      data-repo="anqixu/ftag2"
      data-branch="master"
      data-sha="7e8510f10ddc527a1f5c0b66b2a40040423f2dd2"
  >

    <input type="hidden" name="nwo" value="anqixu/ftag2" />

    <div class="select-menu js-menu-container js-select-menu search-context-select-menu">
      <span class="minibutton select-menu-button js-menu-target" role="button" aria-haspopup="true">
        <span class="js-select-button">This repository</span>
      </span>

      <div class="select-menu-modal-holder js-menu-content js-navigation-container" aria-hidden="true">
        <div class="select-menu-modal">

          <div class="select-menu-item js-navigation-item js-this-repository-navigation-item selected">
            <span class="select-menu-item-icon octicon octicon-check"></span>
            <input type="radio" class="js-search-this-repository" name="search_target" value="repository" checked="checked" />
            <div class="select-menu-item-text js-select-button-text">This repository</div>
          </div> <!-- /.select-menu-item -->

          <div class="select-menu-item js-navigation-item js-all-repositories-navigation-item">
            <span class="select-menu-item-icon octicon octicon-check"></span>
            <input type="radio" name="search_target" value="global" />
            <div class="select-menu-item-text js-select-button-text">All repositories</div>
          </div> <!-- /.select-menu-item -->

        </div>
      </div>
    </div>

  <span class="help tooltipped tooltipped-s" aria-label="Show command bar help">
    <span class="octicon octicon-question"></span>
  </span>


  <input type="hidden" name="ref" value="cmdform">

</form>
        <ul class="top-nav">
          <li class="explore"><a href="/explore">Explore</a></li>
            <li><a href="https://gist.github.com">Gist</a></li>
            <li><a href="/blog">Blog</a></li>
          <li><a href="https://help.github.com">Help</a></li>
        </ul>
      </div>

    


  <ul id="user-links">
    <li>
      <a href="/dacocp" class="name">
        <img alt="David Cortes" class=" js-avatar" data-user="6013159" height="20" src="https://avatars.githubusercontent.com/u/6013159" width="20" /> dacocp
      </a>
    </li>

    <li class="new-menu dropdown-toggle js-menu-container">
      <a href="#" class="js-menu-target tooltipped tooltipped-s" aria-label="Create new...">
        <span class="octicon octicon-plus"></span>
        <span class="dropdown-arrow"></span>
      </a>

      <div class="new-menu-content js-menu-content">
      </div>
    </li>

    <li>
      <a href="/settings/profile" id="account_settings"
        class="tooltipped tooltipped-s"
        aria-label="Account settings ">
        <span class="octicon octicon-tools"></span>
      </a>
    </li>
    <li>
      <a class="tooltipped tooltipped-s" href="/logout" data-method="post" id="logout" aria-label="Sign out">
        <span class="octicon octicon-log-out"></span>
      </a>
    </li>

  </ul>

<div class="js-new-dropdown-contents hidden">
  

<ul class="dropdown-menu">
  <li>
    <a href="/new"><span class="octicon octicon-repo-create"></span> New repository</a>
  </li>
  <li>
    <a href="/organizations/new"><span class="octicon octicon-organization"></span> New organization</a>
  </li>


    <li class="section-title">
      <span title="anqixu/ftag2">This repository</span>
    </li>
      <li>
        <a href="/anqixu/ftag2/issues/new"><span class="octicon octicon-issue-opened"></span> New issue</a>
      </li>
</ul>

</div>


    
  </div>
</div>

      

        




          <div class="site" itemscope itemtype="http://schema.org/WebPage">
    
    <div class="pagehead repohead instapaper_ignore readability-menu">
      <div class="container">
        

<ul class="pagehead-actions">

    <li class="subscription">
      <form accept-charset="UTF-8" action="/notifications/subscribe" class="js-social-container" data-autosubmit="true" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="yzEaw0uwT2crelLCGD7HTJ5I6A/Uxr1ixBFeGkzqFJE=" /></div>  <input id="repository_id" name="repository_id" type="hidden" value="13760217" />

    <div class="select-menu js-menu-container js-select-menu">
      <a class="social-count js-social-count" href="/anqixu/ftag2/watchers">
        2
      </a>
      <span class="minibutton select-menu-button with-count js-menu-target" role="button" tabindex="0" aria-haspopup="true">
        <span class="js-select-button">
          <span class="octicon octicon-eye-unwatch"></span>
          Unwatch
        </span>
      </span>

      <div class="select-menu-modal-holder">
        <div class="select-menu-modal subscription-menu-modal js-menu-content" aria-hidden="true">
          <div class="select-menu-header">
            <span class="select-menu-title">Notification status</span>
            <span class="octicon octicon-remove-close js-menu-close"></span>
          </div> <!-- /.select-menu-header -->

          <div class="select-menu-list js-navigation-container" role="menu">

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_included" name="do" type="radio" value="included" />
                <h4>Not watching</h4>
                <span class="description">You only receive notifications for conversations in which you participate or are @mentioned.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye-watch"></span>
                  Watch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item selected" role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input checked="checked" id="do_subscribed" name="do" type="radio" value="subscribed" />
                <h4>Watching</h4>
                <span class="description">You receive notifications for all conversations in this repository.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye-unwatch"></span>
                  Unwatch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_ignore" name="do" type="radio" value="ignore" />
                <h4>Ignoring</h4>
                <span class="description">You do not receive any notifications for conversations in this repository.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-mute"></span>
                  Stop ignoring
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

          </div> <!-- /.select-menu-list -->

        </div> <!-- /.select-menu-modal -->
      </div> <!-- /.select-menu-modal-holder -->
    </div> <!-- /.select-menu -->

</form>
    </li>

  <li>
  

  <div class="js-toggler-container js-social-container starring-container on">
    <a href="/anqixu/ftag2/unstar"
      class="minibutton with-count js-toggler-target star-button starred"
      aria-label="Unstar this repository" data-remote="true" data-method="post" rel="nofollow">
      <span class="octicon octicon-star-delete"></span><span class="text">Unstar</span>
    </a>

    <a href="/anqixu/ftag2/star"
      class="minibutton with-count js-toggler-target star-button unstarred"
      aria-label="Star this repository" data-remote="true" data-method="post" rel="nofollow">
      <span class="octicon octicon-star"></span><span class="text">Star</span>
    </a>

      <a class="social-count js-social-count" href="/anqixu/ftag2/stargazers">
        2
      </a>
  </div>

  </li>


        <li>
          <a href="/anqixu/ftag2/fork" class="minibutton with-count js-toggler-target fork-button lighter tooltipped-n" title="Fork this repo" rel="nofollow" data-method="post">
            <span class="octicon octicon-git-branch-create"></span><span class="text">Fork</span>
          </a>
          <a href="/anqixu/ftag2/network" class="social-count">0</a>
        </li>


</ul>

        <h1 itemscope itemtype="http://data-vocabulary.org/Breadcrumb" class="entry-title private">
          <span class="repo-label"><span>private</span></span>
          <span class="mega-octicon octicon-lock"></span>
          <span class="author">
            <a href="/anqixu" class="url fn" itemprop="url" rel="author"><span itemprop="title">anqixu</span></a>
          </span>
          <span class="repohead-name-divider">/</span>
          <strong><a href="/anqixu/ftag2" class="js-current-repository js-repo-home-link">ftag2</a></strong>

          <span class="page-context-loader">
            <img alt="Octocat-spinner-32" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
          </span>

        </h1>
      </div><!-- /.container -->
    </div><!-- /.repohead -->

    <div class="container">
      <div class="repository-with-sidebar repo-container new-discussion-timeline js-new-discussion-timeline  ">
        <div class="repository-sidebar clearfix">
            

<div class="sunken-menu vertical-right repo-nav js-repo-nav js-repository-container-pjax js-octicon-loaders">
  <div class="sunken-menu-contents">
    <ul class="sunken-menu-group">
      <li class="tooltipped tooltipped-w" aria-label="Code">
        <a href="/anqixu/ftag2" aria-label="Code" class="selected js-selected-navigation-item sunken-menu-item" data-gotokey="c" data-pjax="true" data-selected-links="repo_source repo_downloads repo_commits repo_tags repo_branches /anqixu/ftag2">
          <span class="octicon octicon-code"></span> <span class="full-word">Code</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

        <li class="tooltipped tooltipped-w" aria-label="Issues">
          <a href="/anqixu/ftag2/issues" aria-label="Issues" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-gotokey="i" data-selected-links="repo_issues /anqixu/ftag2/issues">
            <span class="octicon octicon-issue-opened"></span> <span class="full-word">Issues</span>
            <span class='counter'>0</span>
            <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>

      <li class="tooltipped tooltipped-w" aria-label="Pull Requests">
        <a href="/anqixu/ftag2/pulls" aria-label="Pull Requests" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-gotokey="p" data-selected-links="repo_pulls /anqixu/ftag2/pulls">
            <span class="octicon octicon-git-pull-request"></span> <span class="full-word">Pull Requests</span>
            <span class='counter'>0</span>
            <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>


        <li class="tooltipped tooltipped-w" aria-label="Wiki">
          <a href="/anqixu/ftag2/wiki" aria-label="Wiki" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="repo_wiki /anqixu/ftag2/wiki">
            <span class="octicon octicon-book"></span> <span class="full-word">Wiki</span>
            <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>
    </ul>
    <div class="sunken-menu-separator"></div>
    <ul class="sunken-menu-group">

      <li class="tooltipped tooltipped-w" aria-label="Pulse">
        <a href="/anqixu/ftag2/pulse" aria-label="Pulse" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="pulse /anqixu/ftag2/pulse">
          <span class="octicon octicon-pulse"></span> <span class="full-word">Pulse</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

      <li class="tooltipped tooltipped-w" aria-label="Graphs">
        <a href="/anqixu/ftag2/graphs" aria-label="Graphs" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="repo_graphs repo_contributors /anqixu/ftag2/graphs">
          <span class="octicon octicon-graph"></span> <span class="full-word">Graphs</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

      <li class="tooltipped tooltipped-w" aria-label="Network">
        <a href="/anqixu/ftag2/network" aria-label="Network" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-selected-links="repo_network /anqixu/ftag2/network">
          <span class="octicon octicon-git-branch"></span> <span class="full-word">Network</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>
    </ul>


  </div>
</div>

              <div class="only-with-full-nav">
                

  

<div class="clone-url open"
  data-protocol-type="http"
  data-url="/users/set_protocol?protocol_selector=http&amp;protocol_type=push">
  <h3><strong>HTTPS</strong> clone URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="https://github.com/anqixu/ftag2.git" readonly="readonly">

    <span aria-label="copy to clipboard" class="js-zeroclipboard url-box-clippy minibutton zeroclipboard-button" data-clipboard-text="https://github.com/anqixu/ftag2.git" data-copied-hint="copied!"><span class="octicon octicon-clippy"></span></span>
  </div>
</div>

  

<div class="clone-url "
  data-protocol-type="ssh"
  data-url="/users/set_protocol?protocol_selector=ssh&amp;protocol_type=push">
  <h3><strong>SSH</strong> clone URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="git@github.com:anqixu/ftag2.git" readonly="readonly">

    <span aria-label="copy to clipboard" class="js-zeroclipboard url-box-clippy minibutton zeroclipboard-button" data-clipboard-text="git@github.com:anqixu/ftag2.git" data-copied-hint="copied!"><span class="octicon octicon-clippy"></span></span>
  </div>
</div>

  

<div class="clone-url "
  data-protocol-type="subversion"
  data-url="/users/set_protocol?protocol_selector=subversion&amp;protocol_type=push">
  <h3><strong>Subversion</strong> checkout URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="https://github.com/anqixu/ftag2" readonly="readonly">

    <span aria-label="copy to clipboard" class="js-zeroclipboard url-box-clippy minibutton zeroclipboard-button" data-clipboard-text="https://github.com/anqixu/ftag2" data-copied-hint="copied!"><span class="octicon octicon-clippy"></span></span>
  </div>
</div>


<p class="clone-options">You can clone with
      <a href="#" class="js-clone-selector" data-protocol="http">HTTPS</a>,
      <a href="#" class="js-clone-selector" data-protocol="ssh">SSH</a>,
      or <a href="#" class="js-clone-selector" data-protocol="subversion">Subversion</a>.
  <span class="help tooltipped tooltipped-n" aria-label="Get help on which URL is right for you.">
    <a href="https://help.github.com/articles/which-remote-url-should-i-use">
    <span class="octicon octicon-question"></span>
    </a>
  </span>
</p>



                <a href="/anqixu/ftag2/archive/master.zip"
                   class="minibutton sidebar-button"
                   title="Download this repository as a zip file"
                   rel="nofollow">
                  <span class="octicon octicon-cloud-download"></span>
                  Download ZIP
                </a>
              </div>
        </div><!-- /.repository-sidebar -->

        <div id="js-repo-pjax-container" class="repository-content context-loader-container" data-pjax-container>
          


<!-- blob contrib key: blob_contributors:v21:249cea2282c3649983136277d7bcb172 -->

<p title="This is a placeholder element" class="js-history-link-replace hidden"></p>

<a href="/anqixu/ftag2/find/master" data-pjax data-hotkey="t" class="js-show-file-finder" style="display:none">Show File Finder</a>

<div class="file-navigation">
  

<div class="select-menu js-menu-container js-select-menu" >
  <span class="minibutton select-menu-button js-menu-target" data-hotkey="w"
    data-master-branch="master"
    data-ref="master"
    role="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
    <span class="octicon octicon-git-branch"></span>
    <i>branch:</i>
    <span class="js-select-button">master</span>
  </span>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax aria-hidden="true">

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <span class="select-menu-title">Switch branches/tags</span>
        <span class="octicon octicon-remove-close js-menu-close"></span>
      </div> <!-- /.select-menu-header -->

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Find or create a branch…" id="context-commitish-filter-field" class="js-filterable-field js-navigation-enable" placeholder="Find or create a branch…">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" class="js-select-menu-tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" class="js-select-menu-tab">Tags</a>
            </li>
          </ul>
        </div><!-- /.select-menu-tabs -->
      </div><!-- /.select-menu-filters -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <div class="select-menu-item js-navigation-item selected">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/anqixu/ftag2/blob/master/CMakeLists.txt"
                 data-name="master"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="master">master</a>
            </div> <!-- /.select-menu-item -->
        </div>

          <form accept-charset="UTF-8" action="/anqixu/ftag2/branches" class="js-create-branch select-menu-item select-menu-new-item-form js-navigation-item js-new-item-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="yzEaw0uwT2crelLCGD7HTJ5I6A/Uxr1ixBFeGkzqFJE=" /></div>
            <span class="octicon octicon-git-branch-create select-menu-item-icon"></span>
            <div class="select-menu-item-text">
              <h4>Create branch: <span class="js-new-item-name"></span></h4>
              <span class="description">from ‘master’</span>
            </div>
            <input type="hidden" name="name" id="name" class="js-new-item-value">
            <input type="hidden" name="branch" id="branch" value="master" />
            <input type="hidden" name="path" id="path" value="CMakeLists.txt" />
          </form> <!-- /.select-menu-item -->

      </div> <!-- /.select-menu-list -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div> <!-- /.select-menu-list -->

    </div> <!-- /.select-menu-modal -->
  </div> <!-- /.select-menu-modal-holder -->
</div> <!-- /.select-menu -->

  <div class="breadcrumb">
    <span class='repo-root js-repo-root'><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/anqixu/ftag2" data-branch="master" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">ftag2</span></a></span></span><span class="separator"> / </span><strong class="final-path">CMakeLists.txt</strong> <span aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="CMakeLists.txt" data-copied-hint="copied!"><span class="octicon octicon-clippy"></span></span>
  </div>
</div>


  <div class="commit file-history-tease">
    <img class="main-avatar" height="24" src="https://2.gravatar.com/avatar/9b1df43546f8089c97fd34f334584da4?d=https%3A%2F%2Fa248.e.akamai.net%2Fassets.github.com%2Fimages%2Fgravatars%2Fgravatar-user-420.png&amp;r=x&amp;s=140" width="24" />
    <span class="author"><span rel="author">Anqi Xu</span></span>
    <time class="js-relative-date" data-title-format="YYYY-MM-DD HH:mm:ss" datetime="2014-03-03T15:59:02-08:00" title="2014-03-03 15:59:02">March 03, 2014</time>
    <div class="commit-title">
        <a href="/anqixu/ftag2/commit/6e89d7b603e8956d7a3538ea4e4d2b1a0014108a" class="message" data-pjax="true" title="enhanced detector logic; cleaned up code a bit">enhanced detector logic; cleaned up code a bit</a>
    </div>

    <div class="participation">
      <p class="quickstat"><a href="#blob_contributors_box" rel="facebox"><strong>1</strong> contributor</a></p>
      
    </div>
    <div id="blob_contributors_box" style="display:none">
      <h2 class="facebox-header">Users who have contributed to this file</h2>
      <ul class="facebox-user-list">
          <li class="facebox-user-list-item">
            <img alt="David Cortes" class=" js-avatar" data-user="6013159" height="24" src="https://2.gravatar.com/avatar/2c4d06ae0480ae1835f1262afb89a364?d=https%3A%2F%2Fidenticons.github.com%2F7c14de49316ff68a794f40de77191f24.png&amp;r=x&amp;s=140" width="24" />
            <a href="/dacocp">dacocp</a>
          </li>
      </ul>
    </div>
  </div>

<div class="file-box">
  <div class="file">
    <div class="meta clearfix">
      <div class="info file-name">
        <span class="icon"><b class="octicon octicon-file-text"></b></span>
        <span class="mode" title="File Mode">file</span>
        <span class="meta-divider"></span>
          <span>181 lines (150 sloc)</span>
          <span class="meta-divider"></span>
        <span>4.953 kb</span>
      </div>
      <div class="actions">
        <div class="button-group">
                <a class="minibutton js-update-url-with-hash"
                   href="/anqixu/ftag2/edit/master/CMakeLists.txt"
                   data-method="post" rel="nofollow" data-hotkey="e">Edit</a>
          <a href="/anqixu/ftag2/raw/master/CMakeLists.txt" class="button minibutton " id="raw-url">Raw</a>
            <a href="/anqixu/ftag2/blame/master/CMakeLists.txt" class="button minibutton js-update-url-with-hash">Blame</a>
          <a href="/anqixu/ftag2/commits/master/CMakeLists.txt" class="button minibutton " rel="nofollow">History</a>
        </div><!-- /.button-group -->
          <a class="minibutton danger empty-icon tooltipped tooltipped-s"
             href="/anqixu/ftag2/delete/master/CMakeLists.txt"
             aria-label=""
             data-method="post" data-test-id="delete-blob-file" rel="nofollow">
          Delete
        </a>
      </div><!-- /.actions -->
    </div>
        <div class="blob-wrapper data type-cmake js-blob-data">
        <table class="file-code file-diff tab-size-8">
          <tr class="file-code-line">
            <td class="blob-line-nums">
              <span id="L1" rel="#L1">1</span>
<span id="L2" rel="#L2">2</span>
<span id="L3" rel="#L3">3</span>
<span id="L4" rel="#L4">4</span>
<span id="L5" rel="#L5">5</span>
<span id="L6" rel="#L6">6</span>
<span id="L7" rel="#L7">7</span>
<span id="L8" rel="#L8">8</span>
<span id="L9" rel="#L9">9</span>
<span id="L10" rel="#L10">10</span>
<span id="L11" rel="#L11">11</span>
<span id="L12" rel="#L12">12</span>
<span id="L13" rel="#L13">13</span>
<span id="L14" rel="#L14">14</span>
<span id="L15" rel="#L15">15</span>
<span id="L16" rel="#L16">16</span>
<span id="L17" rel="#L17">17</span>
<span id="L18" rel="#L18">18</span>
<span id="L19" rel="#L19">19</span>
<span id="L20" rel="#L20">20</span>
<span id="L21" rel="#L21">21</span>
<span id="L22" rel="#L22">22</span>
<span id="L23" rel="#L23">23</span>
<span id="L24" rel="#L24">24</span>
<span id="L25" rel="#L25">25</span>
<span id="L26" rel="#L26">26</span>
<span id="L27" rel="#L27">27</span>
<span id="L28" rel="#L28">28</span>
<span id="L29" rel="#L29">29</span>
<span id="L30" rel="#L30">30</span>
<span id="L31" rel="#L31">31</span>
<span id="L32" rel="#L32">32</span>
<span id="L33" rel="#L33">33</span>
<span id="L34" rel="#L34">34</span>
<span id="L35" rel="#L35">35</span>
<span id="L36" rel="#L36">36</span>
<span id="L37" rel="#L37">37</span>
<span id="L38" rel="#L38">38</span>
<span id="L39" rel="#L39">39</span>
<span id="L40" rel="#L40">40</span>
<span id="L41" rel="#L41">41</span>
<span id="L42" rel="#L42">42</span>
<span id="L43" rel="#L43">43</span>
<span id="L44" rel="#L44">44</span>
<span id="L45" rel="#L45">45</span>
<span id="L46" rel="#L46">46</span>
<span id="L47" rel="#L47">47</span>
<span id="L48" rel="#L48">48</span>
<span id="L49" rel="#L49">49</span>
<span id="L50" rel="#L50">50</span>
<span id="L51" rel="#L51">51</span>
<span id="L52" rel="#L52">52</span>
<span id="L53" rel="#L53">53</span>
<span id="L54" rel="#L54">54</span>
<span id="L55" rel="#L55">55</span>
<span id="L56" rel="#L56">56</span>
<span id="L57" rel="#L57">57</span>
<span id="L58" rel="#L58">58</span>
<span id="L59" rel="#L59">59</span>
<span id="L60" rel="#L60">60</span>
<span id="L61" rel="#L61">61</span>
<span id="L62" rel="#L62">62</span>
<span id="L63" rel="#L63">63</span>
<span id="L64" rel="#L64">64</span>
<span id="L65" rel="#L65">65</span>
<span id="L66" rel="#L66">66</span>
<span id="L67" rel="#L67">67</span>
<span id="L68" rel="#L68">68</span>
<span id="L69" rel="#L69">69</span>
<span id="L70" rel="#L70">70</span>
<span id="L71" rel="#L71">71</span>
<span id="L72" rel="#L72">72</span>
<span id="L73" rel="#L73">73</span>
<span id="L74" rel="#L74">74</span>
<span id="L75" rel="#L75">75</span>
<span id="L76" rel="#L76">76</span>
<span id="L77" rel="#L77">77</span>
<span id="L78" rel="#L78">78</span>
<span id="L79" rel="#L79">79</span>
<span id="L80" rel="#L80">80</span>
<span id="L81" rel="#L81">81</span>
<span id="L82" rel="#L82">82</span>
<span id="L83" rel="#L83">83</span>
<span id="L84" rel="#L84">84</span>
<span id="L85" rel="#L85">85</span>
<span id="L86" rel="#L86">86</span>
<span id="L87" rel="#L87">87</span>
<span id="L88" rel="#L88">88</span>
<span id="L89" rel="#L89">89</span>
<span id="L90" rel="#L90">90</span>
<span id="L91" rel="#L91">91</span>
<span id="L92" rel="#L92">92</span>
<span id="L93" rel="#L93">93</span>
<span id="L94" rel="#L94">94</span>
<span id="L95" rel="#L95">95</span>
<span id="L96" rel="#L96">96</span>
<span id="L97" rel="#L97">97</span>
<span id="L98" rel="#L98">98</span>
<span id="L99" rel="#L99">99</span>
<span id="L100" rel="#L100">100</span>
<span id="L101" rel="#L101">101</span>
<span id="L102" rel="#L102">102</span>
<span id="L103" rel="#L103">103</span>
<span id="L104" rel="#L104">104</span>
<span id="L105" rel="#L105">105</span>
<span id="L106" rel="#L106">106</span>
<span id="L107" rel="#L107">107</span>
<span id="L108" rel="#L108">108</span>
<span id="L109" rel="#L109">109</span>
<span id="L110" rel="#L110">110</span>
<span id="L111" rel="#L111">111</span>
<span id="L112" rel="#L112">112</span>
<span id="L113" rel="#L113">113</span>
<span id="L114" rel="#L114">114</span>
<span id="L115" rel="#L115">115</span>
<span id="L116" rel="#L116">116</span>
<span id="L117" rel="#L117">117</span>
<span id="L118" rel="#L118">118</span>
<span id="L119" rel="#L119">119</span>
<span id="L120" rel="#L120">120</span>
<span id="L121" rel="#L121">121</span>
<span id="L122" rel="#L122">122</span>
<span id="L123" rel="#L123">123</span>
<span id="L124" rel="#L124">124</span>
<span id="L125" rel="#L125">125</span>
<span id="L126" rel="#L126">126</span>
<span id="L127" rel="#L127">127</span>
<span id="L128" rel="#L128">128</span>
<span id="L129" rel="#L129">129</span>
<span id="L130" rel="#L130">130</span>
<span id="L131" rel="#L131">131</span>
<span id="L132" rel="#L132">132</span>
<span id="L133" rel="#L133">133</span>
<span id="L134" rel="#L134">134</span>
<span id="L135" rel="#L135">135</span>
<span id="L136" rel="#L136">136</span>
<span id="L137" rel="#L137">137</span>
<span id="L138" rel="#L138">138</span>
<span id="L139" rel="#L139">139</span>
<span id="L140" rel="#L140">140</span>
<span id="L141" rel="#L141">141</span>
<span id="L142" rel="#L142">142</span>
<span id="L143" rel="#L143">143</span>
<span id="L144" rel="#L144">144</span>
<span id="L145" rel="#L145">145</span>
<span id="L146" rel="#L146">146</span>
<span id="L147" rel="#L147">147</span>
<span id="L148" rel="#L148">148</span>
<span id="L149" rel="#L149">149</span>
<span id="L150" rel="#L150">150</span>
<span id="L151" rel="#L151">151</span>
<span id="L152" rel="#L152">152</span>
<span id="L153" rel="#L153">153</span>
<span id="L154" rel="#L154">154</span>
<span id="L155" rel="#L155">155</span>
<span id="L156" rel="#L156">156</span>
<span id="L157" rel="#L157">157</span>
<span id="L158" rel="#L158">158</span>
<span id="L159" rel="#L159">159</span>
<span id="L160" rel="#L160">160</span>
<span id="L161" rel="#L161">161</span>
<span id="L162" rel="#L162">162</span>
<span id="L163" rel="#L163">163</span>
<span id="L164" rel="#L164">164</span>
<span id="L165" rel="#L165">165</span>
<span id="L166" rel="#L166">166</span>
<span id="L167" rel="#L167">167</span>
<span id="L168" rel="#L168">168</span>
<span id="L169" rel="#L169">169</span>
<span id="L170" rel="#L170">170</span>
<span id="L171" rel="#L171">171</span>
<span id="L172" rel="#L172">172</span>
<span id="L173" rel="#L173">173</span>
<span id="L174" rel="#L174">174</span>
<span id="L175" rel="#L175">175</span>
<span id="L176" rel="#L176">176</span>
<span id="L177" rel="#L177">177</span>
<span id="L178" rel="#L178">178</span>
<span id="L179" rel="#L179">179</span>
<span id="L180" rel="#L180">180</span>

            </td>
            <td class="blob-line-code"><div class="code-body highlight"><pre><div class='line' id='LC1'><span class="nb">set</span><span class="p">(</span><span class="s">FTAG2_LIB_NAME</span> <span class="s2">&quot;ftag2&quot;</span> <span class="s">CACHE</span> <span class="s">STRING</span> <span class="s2">&quot;FTag v2 API library name&quot;</span><span class="p">)</span></div><div class='line' id='LC2'><span class="nb">set</span><span class="p">(</span><span class="s">FTAG2_READER_NODELET_NAME</span> <span class="s2">&quot;ftag2_reader_nodelet&quot;</span><span class="p">)</span></div><div class='line' id='LC3'><br/></div><div class='line' id='LC4'><span class="nb">cmake_minimum_required</span><span class="p">(</span><span class="s">VERSION</span> <span class="s">2.8.3</span><span class="p">)</span></div><div class='line' id='LC5'><span class="nb">project</span><span class="p">(</span><span class="s">ftag2</span><span class="p">)</span></div><div class='line' id='LC6'><br/></div><div class='line' id='LC7'><span class="nb">set</span><span class="p">(</span><span class="s">CMAKE_CXX_FLAGS</span> <span class="s2">&quot;-std=c++0x ${CMAKE_CXX_FLAGS} -g -Wall -O2&quot;</span><span class="p">)</span></div><div class='line' id='LC8'><br/></div><div class='line' id='LC9'><span class="nb">find_package</span><span class="p">(</span><span class="s">catkin</span> <span class="s">REQUIRED</span> <span class="s">COMPONENTS</span></div><div class='line' id='LC10'>&nbsp;&nbsp;<span class="s">roscpp</span></div><div class='line' id='LC11'>&nbsp;&nbsp;<span class="s">rospy</span></div><div class='line' id='LC12'>&nbsp;&nbsp;<span class="s">nodelet</span></div><div class='line' id='LC13'>&nbsp;&nbsp;<span class="s">dynamic_reconfigure</span></div><div class='line' id='LC14'>&nbsp;&nbsp;<span class="s">image_transport</span></div><div class='line' id='LC15'>&nbsp;&nbsp;<span class="s">std_msgs</span></div><div class='line' id='LC16'>&nbsp;&nbsp;<span class="s">cv_bridge</span></div><div class='line' id='LC17'>&nbsp;&nbsp;<span class="s">geometry_msgs</span></div><div class='line' id='LC18'>&nbsp;&nbsp;<span class="s">sensor_msgs</span></div><div class='line' id='LC19'>&nbsp;&nbsp;<span class="s">tf</span></div><div class='line' id='LC20'>&nbsp;&nbsp;<span class="s">image_geometry</span></div><div class='line' id='LC21'>&nbsp;&nbsp;<span class="s">image_transport</span></div><div class='line' id='LC22'><span class="p">)</span></div><div class='line' id='LC23'><br/></div><div class='line' id='LC24'><span class="nb">find_package</span><span class="p">(</span><span class="s">OMPL</span> <span class="s">REQUIRED</span><span class="p">)</span></div><div class='line' id='LC25'><br/></div><div class='line' id='LC26'><span class="nb">find_package</span><span class="p">(</span><span class="s">Boost</span> <span class="s">REQUIRED</span> <span class="s">COMPONENTS</span> <span class="s">date_time-mt</span> <span class="s">system-mt</span> <span class="s">thread-mt</span> <span class="s">serialization-mt</span> <span class="s">program_options-mt</span><span class="p">)</span></div><div class='line' id='LC27'><span class="nb">find_package</span><span class="p">(</span><span class="s">OpenCV</span> <span class="s">REQUIRED</span> <span class="s">COMPONENTS</span> <span class="s">core</span> <span class="s">imgproc</span> <span class="s">highgui</span> <span class="s">calib3d</span><span class="p">)</span></div><div class='line' id='LC28'><br/></div><div class='line' id='LC29'><span class="c">#######################################</span></div><div class='line' id='LC30'><span class="c">## Declare ROS messages and services ##</span></div><div class='line' id='LC31'><span class="c">#######################################</span></div><div class='line' id='LC32'><br/></div><div class='line' id='LC33'><span class="nb">add_message_files</span><span class="p">(</span></div><div class='line' id='LC34'>&nbsp;&nbsp;<span class="s">DIRECTORY</span> <span class="s">msg</span></div><div class='line' id='LC35'>&nbsp;&nbsp;<span class="s">FILES</span> <span class="s">FreqTBMarkerInfo.msg</span> <span class="s">FreqTBPhaseStats.msg</span> <span class="s">TagDetection.msg</span> <span class="s">TagDetections.msg</span></div><div class='line' id='LC36'><span class="p">)</span></div><div class='line' id='LC37'><br/></div><div class='line' id='LC38'><span class="c">## Generate services in the &#39;srv&#39; folder</span></div><div class='line' id='LC39'><span class="c"># add_service_files(</span></div><div class='line' id='LC40'><span class="c">#   FILES</span></div><div class='line' id='LC41'><span class="c">#   Service1.srv</span></div><div class='line' id='LC42'><span class="c"># )</span></div><div class='line' id='LC43'><br/></div><div class='line' id='LC44'><span class="c">## Generate added messages and services with any dependencies listed here</span></div><div class='line' id='LC45'><span class="nb">generate_messages</span><span class="p">(</span></div><div class='line' id='LC46'>&nbsp;&nbsp;<span class="s">DEPENDENCIES</span></div><div class='line' id='LC47'>&nbsp;&nbsp;<span class="s">geometry_msgs</span> <span class="s">sensor_msgs</span> <span class="s">std_msgs</span></div><div class='line' id='LC48'><span class="p">)</span></div><div class='line' id='LC49'><br/></div><div class='line' id='LC50'><span class="nb">generate_dynamic_reconfigure_options</span><span class="p">(</span></div><div class='line' id='LC51'>&nbsp;&nbsp;<span class="s">cfg/FTag2Reader.cfg</span></div><div class='line' id='LC52'>&nbsp;&nbsp;<span class="s">cfg/CamTestbench.cfg</span></div><div class='line' id='LC53'>&nbsp;&nbsp;<span class="s">cfg/FreqTestbench.cfg</span></div><div class='line' id='LC54'>&nbsp;&nbsp;<span class="p">)</span></div><div class='line' id='LC55'><br/></div><div class='line' id='LC56'><br/></div><div class='line' id='LC57'><span class="c">###################################</span></div><div class='line' id='LC58'><span class="c">## catkin specific configuration ##</span></div><div class='line' id='LC59'><span class="c">###################################</span></div><div class='line' id='LC60'><span class="c">## The catkin_package macro generates cmake config files for your package</span></div><div class='line' id='LC61'><span class="c">## Declare things to be passed to dependent projects</span></div><div class='line' id='LC62'><span class="c">## INCLUDE_DIRS: uncomment this if you package contains header files</span></div><div class='line' id='LC63'><span class="c">## LIBRARIES: libraries you create in this project that dependent projects also need</span></div><div class='line' id='LC64'><span class="c">## CATKIN_DEPENDS: catkin_packages dependent projects also need</span></div><div class='line' id='LC65'><span class="c">## DEPENDS: system dependencies of this project that dependent projects also need</span></div><div class='line' id='LC66'><span class="nb">catkin_package</span><span class="p">(</span></div><div class='line' id='LC67'>&nbsp;&nbsp;<span class="s">INCLUDE_DIRS</span> <span class="s">include</span></div><div class='line' id='LC68'>&nbsp;&nbsp;<span class="s">LIBRARIES</span> <span class="o">${</span><span class="nv">FTAG2_LIB_NAME</span><span class="o">}</span> <span class="o">${</span><span class="nv">FTAG2_READER_NODELET_NAME</span><span class="o">}</span></div><div class='line' id='LC69'>&nbsp;&nbsp;<span class="s">CATKIN_DEPENDS</span> <span class="s">roscpp</span> <span class="s">rospy</span> <span class="s">nodelet</span> <span class="s">dynamic_reconfigure</span> <span class="s">image_transport</span> <span class="s">cv_bridge</span> <span class="s">std_msgs</span> <span class="s">geometry_msgs</span> <span class="s">sensor_msgs</span> <span class="s">tf</span> </div><div class='line' id='LC70'><span class="c">#  DEPENDS system_lib</span></div><div class='line' id='LC71'><span class="p">)</span></div><div class='line' id='LC72'><br/></div><div class='line' id='LC73'><span class="c">###########</span></div><div class='line' id='LC74'><span class="c">## Build ##</span></div><div class='line' id='LC75'><span class="c">###########</span></div><div class='line' id='LC76'><br/></div><div class='line' id='LC77'><span class="nb">include_directories</span><span class="p">(</span></div><div class='line' id='LC78'>&nbsp;&nbsp;<span class="s">include</span></div><div class='line' id='LC79'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">catkin_INCLUDE_DIRS</span><span class="o">}</span></div><div class='line' id='LC80'><span class="p">)</span></div><div class='line' id='LC81'><span class="nb">link_directories</span><span class="p">(</span></div><div class='line' id='LC82'>&nbsp;&nbsp;<span class="s">lib</span></div><div class='line' id='LC83'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">OpenCV_LIBRARY_DIRS</span><span class="o">}</span></div><div class='line' id='LC84'><span class="p">)</span></div><div class='line' id='LC85'><br/></div><div class='line' id='LC86'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">COMMON_HEADERS</span>   <span class="s2">&quot;include/common/*.h*&quot;</span> <span class="p">)</span></div><div class='line' id='LC87'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">DETECTOR_HEADERS</span> <span class="s2">&quot;include/detector/*.h*&quot;</span> <span class="p">)</span></div><div class='line' id='LC88'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">DECODER_HEADERS</span>  <span class="s2">&quot;include/decoder/*.h*&quot;</span> <span class="p">)</span></div><div class='line' id='LC89'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">ENCODER_HEADERS</span>  <span class="s2">&quot;include/encoder/*.h*&quot;</span> <span class="p">)</span></div><div class='line' id='LC90'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">TRACKER_HEADERS</span>  <span class="s2">&quot;include/tracker/*.h*&quot;</span> <span class="p">)</span></div><div class='line' id='LC91'><br/></div><div class='line' id='LC92'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">COMMON_SOURCES</span>   <span class="s2">&quot;src/common/*.c*&quot;</span> <span class="p">)</span></div><div class='line' id='LC93'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">DETECTOR_SOURCES</span> <span class="s2">&quot;src/detector/*.c*&quot;</span> <span class="p">)</span></div><div class='line' id='LC94'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">DECODER_SOURCES</span>  <span class="s2">&quot;src/decoder/*.c*&quot;</span> <span class="p">)</span></div><div class='line' id='LC95'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">ENCODER_SOURCES</span>  <span class="s2">&quot;src/encoder/*.c*&quot;</span> <span class="p">)</span></div><div class='line' id='LC96'><span class="nb">FILE</span> <span class="p">(</span> <span class="s">GLOB</span> <span class="s">TRACKER_SOURCES</span>  <span class="s2">&quot;src/tracker/*.c*&quot;</span> <span class="p">)</span></div><div class='line' id='LC97'><br/></div><div class='line' id='LC98'><span class="nb">add_library</span><span class="p">(</span><span class="o">${</span><span class="nv">FTAG2_LIB_NAME</span><span class="o">}</span></div><div class='line' id='LC99'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">COMMON_SOURCES</span><span class="o">}</span></div><div class='line' id='LC100'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">DETECTOR_SOURCES</span><span class="o">}</span></div><div class='line' id='LC101'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">DECODER_SOURCES</span><span class="o">}</span></div><div class='line' id='LC102'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">ENCODER_SOURCES</span><span class="o">}</span></div><div class='line' id='LC103'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">TRACKER_SOURCES</span><span class="o">}</span></div><div class='line' id='LC104'><span class="p">)</span></div><div class='line' id='LC105'><span class="nb">target_link_libraries</span><span class="p">(</span> <span class="o">${</span><span class="nv">FTAG2_LIB_NAME</span><span class="o">}</span></div><div class='line' id='LC106'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">OMPL_LIBRARIES</span><span class="o">}</span></div><div class='line' id='LC107'><span class="c">#  ${OpenCV_LIBRARIES}</span></div><div class='line' id='LC108'><span class="p">)</span></div><div class='line' id='LC109'><br/></div><div class='line' id='LC110'><span class="nb">add_library</span><span class="p">(</span><span class="o">${</span><span class="nv">FTAG2_READER_NODELET_NAME</span><span class="o">}</span> <span class="s">src/ftag2_reader_nodelet.cpp</span><span class="p">)</span></div><div class='line' id='LC111'><span class="nb">target_link_libraries</span><span class="p">(</span><span class="o">${</span><span class="nv">FTAG2_READER_NODELET_NAME</span><span class="o">}</span></div><div class='line' id='LC112'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">FTAG2_LIB_NAME</span><span class="o">}</span></div><div class='line' id='LC113'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">OpenCV_LIBRARIES</span><span class="o">}</span></div><div class='line' id='LC114'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">catkin_LIBRARIES</span><span class="o">}</span></div><div class='line' id='LC115'><span class="p">)</span></div><div class='line' id='LC116'><span class="nb">add_dependencies</span><span class="p">(</span><span class="o">${</span><span class="nv">FTAG2_READER_NODELET_NAME</span><span class="o">}</span> <span class="o">${</span><span class="nv">PROJECT_NAME</span><span class="o">}</span><span class="s">_gencfg</span><span class="p">)</span></div><div class='line' id='LC117'><br/></div><div class='line' id='LC118'><span class="nb">add_executable</span><span class="p">(</span><span class="s">ros_cam_testbench</span> <span class="s">src/ros_cam_testbench.cpp</span><span class="p">)</span></div><div class='line' id='LC119'><span class="nb">target_link_libraries</span><span class="p">(</span><span class="s">ros_cam_testbench</span></div><div class='line' id='LC120'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">FTAG2_LIB_NAME</span><span class="o">}</span></div><div class='line' id='LC121'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">OpenCV_LIBRARIES</span><span class="o">}</span></div><div class='line' id='LC122'>&nbsp;&nbsp;<span class="o">${</span><span class="nv">catkin_LIBRARIES</span><span class="o">}</span></div><div class='line' id='LC123'>&nbsp;&nbsp;<span class="s">yaml-cpp</span></div><div class='line' id='LC124'>&nbsp;<span class="p">)</span></div><div class='line' id='LC125'><span class="nb">add_dependencies</span><span class="p">(</span><span class="s">ros_cam_testbench</span> <span class="o">${</span><span class="nv">PROJECT_NAME</span><span class="o">}</span><span class="s">_gencfg</span><span class="p">)</span></div><div class='line' id='LC126'><br/></div><div class='line' id='LC127'><span class="c">#add_executable(freq_testbench src/freq_testbench.cpp)</span></div><div class='line' id='LC128'><span class="c">#target_link_libraries(freq_testbench</span></div><div class='line' id='LC129'><span class="c">#  ${FTAG2_LIB_NAME}</span></div><div class='line' id='LC130'><span class="c">#  ${OpenCV_LIBRARIES}</span></div><div class='line' id='LC131'><span class="c">#  ${catkin_LIBRARIES}</span></div><div class='line' id='LC132'><span class="c">#)</span></div><div class='line' id='LC133'><span class="c">#add_dependencies(freq_testbench ${PROJECT_NAME}_gencfg ftag2_generate_messages_cpp)</span></div><div class='line' id='LC134'><br/></div><div class='line' id='LC135'><br/></div><div class='line' id='LC136'><span class="c">#############</span></div><div class='line' id='LC137'><span class="c">## Install ##</span></div><div class='line' id='LC138'><span class="c">#############</span></div><div class='line' id='LC139'><br/></div><div class='line' id='LC140'><span class="c">## Mark executable scripts (Python etc.) for installation</span></div><div class='line' id='LC141'><span class="c">## in contrast to setup.py, you can choose the destination</span></div><div class='line' id='LC142'><span class="c"># install(PROGRAMS</span></div><div class='line' id='LC143'><span class="c">#   scripts/my_python_script</span></div><div class='line' id='LC144'><span class="c">#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}</span></div><div class='line' id='LC145'><span class="c"># )</span></div><div class='line' id='LC146'><br/></div><div class='line' id='LC147'><span class="c">## Mark executables and/or libraries for installation</span></div><div class='line' id='LC148'><span class="nb">install</span><span class="p">(</span><span class="s">TARGETS</span> <span class="o">${</span><span class="nv">FTAG2_LIB_NAME</span><span class="o">}</span> <span class="o">${</span><span class="nv">FTAG2_READER_NODELET_NAME</span><span class="o">}</span></div><div class='line' id='LC149'>&nbsp;&nbsp;<span class="s">ARCHIVE</span> <span class="s">DESTINATION</span> <span class="o">${</span><span class="nv">CATKIN_PACKAGE_LIB_DESTINATION</span><span class="o">}</span></div><div class='line' id='LC150'>&nbsp;&nbsp;<span class="s">LIBRARY</span> <span class="s">DESTINATION</span> <span class="o">${</span><span class="nv">CATKIN_PACKAGE_LIB_DESTINATION</span><span class="o">}</span></div><div class='line' id='LC151'>&nbsp;&nbsp;<span class="s">RUNTIME</span> <span class="s">DESTINATION</span> <span class="o">${</span><span class="nv">CATKIN_PACKAGE_BIN_DESTINATION</span><span class="o">}</span></div><div class='line' id='LC152'>&nbsp;&nbsp;<span class="p">)</span></div><div class='line' id='LC153'><br/></div><div class='line' id='LC154'><span class="c">## Mark cpp header files for installation</span></div><div class='line' id='LC155'><span class="c"># install(DIRECTORY include/${PROJECT_NAME}/</span></div><div class='line' id='LC156'><span class="c">#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}</span></div><div class='line' id='LC157'><span class="c">#   FILES_MATCHING PATTERN &quot;*.h&quot;</span></div><div class='line' id='LC158'><span class="c">#   PATTERN &quot;.svn&quot; EXCLUDE</span></div><div class='line' id='LC159'><span class="c"># )</span></div><div class='line' id='LC160'><br/></div><div class='line' id='LC161'><span class="nb">install</span><span class="p">(</span><span class="s">FILES</span> <span class="s">nodelet_plugins.xml</span></div><div class='line' id='LC162'>&nbsp;&nbsp;<span class="s">DESTINATION</span> <span class="o">${</span><span class="nv">CATKIN_PACKAGE_SHARE_DESTINATION</span><span class="o">}</span></div><div class='line' id='LC163'>&nbsp;&nbsp;<span class="p">)</span></div><div class='line' id='LC164'><br/></div><div class='line' id='LC165'><span class="nb">install</span><span class="p">(</span><span class="s">DIRECTORY</span> <span class="s">launch/</span></div><div class='line' id='LC166'>&nbsp;&nbsp;<span class="s">DESTINATION</span> <span class="o">${</span><span class="nv">CATKIN_PACKAGE_SHARE_DESTINATION</span><span class="o">}</span><span class="s">/launch</span></div><div class='line' id='LC167'>&nbsp;&nbsp;<span class="p">)</span></div><div class='line' id='LC168'><br/></div><div class='line' id='LC169'><span class="c">#############</span></div><div class='line' id='LC170'><span class="c">## Testing ##</span></div><div class='line' id='LC171'><span class="c">#############</span></div><div class='line' id='LC172'><br/></div><div class='line' id='LC173'><span class="c">## Add gtest based cpp test target and link libraries</span></div><div class='line' id='LC174'><span class="c"># catkin_add_gtest(${PROJECT_NAME}-test test/test_ftag2.cpp)</span></div><div class='line' id='LC175'><span class="c"># if(TARGET ${PROJECT_NAME}-test)</span></div><div class='line' id='LC176'><span class="c">#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})</span></div><div class='line' id='LC177'><span class="c"># endif()</span></div><div class='line' id='LC178'><br/></div><div class='line' id='LC179'><span class="c">## Add folders to be run by python nosetests</span></div><div class='line' id='LC180'><span class="c"># catkin_add_nosetests(test)</span></div></pre></div></td>
          </tr>
        </table>
  </div>

  </div>
</div>

<a href="#jump-to-line" rel="facebox[.linejump]" data-hotkey="l" class="js-jump-to-line" style="display:none">Jump to Line</a>
<div id="jump-to-line" style="display:none">
  <form accept-charset="UTF-8" class="js-jump-to-line-form">
    <input class="linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" autofocus>
    <button type="submit" class="button">Go</button>
  </form>
</div>

        </div>

      </div><!-- /.repo-container -->
      <div class="modal-backdrop"></div>
    </div><!-- /.container -->
  </div><!-- /.site -->


    </div><!-- /.wrapper -->

      <div class="container">
  <div class="site-footer">
    <ul class="site-footer-links right">
      <li><a href="https://status.github.com/">Status</a></li>
      <li><a href="http://developer.github.com">API</a></li>
      <li><a href="http://training.github.com">Training</a></li>
      <li><a href="http://shop.github.com">Shop</a></li>
      <li><a href="/blog">Blog</a></li>
      <li><a href="/about">About</a></li>

    </ul>

    <a href="/">
      <span class="mega-octicon octicon-mark-github" title="GitHub"></span>
    </a>

    <ul class="site-footer-links">
      <li>&copy; 2014 <span title="0.05471s from github-fe140-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
        <li><a href="/site/terms">Terms</a></li>
        <li><a href="/site/privacy">Privacy</a></li>
        <li><a href="/security">Security</a></li>
        <li><a href="/contact">Contact</a></li>
    </ul>
  </div><!-- /.site-footer -->
</div><!-- /.container -->


    <div class="fullscreen-overlay js-fullscreen-overlay" id="fullscreen_overlay">
  <div class="fullscreen-container js-fullscreen-container">
    <div class="textarea-wrap">
      <textarea name="fullscreen-contents" id="fullscreen-contents" class="js-fullscreen-contents" placeholder="" data-suggester="fullscreen_suggester"></textarea>
    </div>
  </div>
  <div class="fullscreen-sidebar">
    <a href="#" class="exit-fullscreen js-exit-fullscreen tooltipped tooltipped-w" aria-label="Exit Zen Mode">
      <span class="mega-octicon octicon-screen-normal"></span>
    </a>
    <a href="#" class="theme-switcher js-theme-switcher tooltipped tooltipped-w"
      aria-label="Switch themes">
      <span class="octicon octicon-color-mode"></span>
    </a>
  </div>
</div>



    <div id="ajax-error-message" class="flash flash-error">
      <span class="octicon octicon-alert"></span>
      <a href="#" class="octicon octicon-remove-close close js-ajax-error-dismiss"></a>
      Something went wrong with that request. Please try again.
    </div>

  </body>
</html>

