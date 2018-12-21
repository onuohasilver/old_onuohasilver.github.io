---
title: "CODEAID: Moving my module to PyPi"
date: 2018-12-20
tags: [machine learning, data science, modules]
header:
excerpt: "Machine Learning, modules, Data Science"
mathjax: "true"
---

# CODEAID
Two things were on my checklist for today, hopefully I wanted
to upload the module that I have been using on some of my data
science projects so that my friends could also benefit from it
and also contribute. Secondly I wanted to create this github.io
page that you are currently viewing now too.
*Yeah, crazy day right?*

So first off I was going from website to website to find the 
exact way to accomplish this and since I have finally achieved it
I think it is only fair that I put together the exact steps I 
followed.


[*The First Link I used, a blog post by jetbrains.*]
(https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/)
I kept on getting an error while following the steps, then I quickly 
went for a quick search on google. 
Found out the link contained in the .pypirc file as indicated in the jetbrain
instructional material was deprecated. So I quickly went over to the python
website and found the current link and kaboom! it just worked like magic
[*Now this fixed it*](https://packaging.python.org/guides/migrating-to-pypi-org/)

Now the codeaid module can be installed by using this simple line of code

```python
    pip install codeaid
```

Currently the codeaid module has just 13 functions but hopefully before a
full month runs out we would have lots of functions that will make
the data science life far much easier for us.