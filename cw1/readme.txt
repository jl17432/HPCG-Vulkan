To switch render mode between normal render mode and mipmap colour mode, 
please press 1 or 2:
	1 stands for normal texture render mode,
	2 stands for mipmap colour render mode

By default, Anisotropic filtering is unabled, 
to enable it please locate line 294 in vkutil.cpp, change "false" to "true".

I had deleted the third_party folder for submition, I didnot add anything in it,
please copy a third_party folder to the project folder so that it will compile