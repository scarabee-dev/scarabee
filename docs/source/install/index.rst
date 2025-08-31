.. _install:

===========================
Installation
===========================

Installation with Pip on Windows, Mac, and Linux
================================================

For most users, it is recommended that you install the latest stable release of
Scarabée using pre-built Python wheels from PyPI using pip. Pre-built wheels
are available for the following operating system - architecture combinations:

- Windows x86
- Linux x86
- macOS x86 (older Intel Macs)
- macOS ARM64 (newer M1 Macs)

If you are not running one of these combinations, you will likely need to
`compile Scarabée on your system from source code <install_source_>`_.

.. Important::

  Scarabée requires that you have **Python 3.9 or later** installed. If you do
  not have a compatible Python version, you will not be able to install
  Scarabée.

To install Scarabée, open a terminal (command prompt on Windows) and enter the
following command:

.. code-block:: bash

   pip install --user scarabee


.. _install_source:

Installation from Source
========================

If you are a developer, want the most recent development version of Scarabée,
or are running on a system for which pre-built wheels do not exist, then you
will have to compile Scarabée from source. Before you can do this, you will
need to ensure that you have installed `CMake <https://cmake.org/>`_ and
`git <https://git-scm.com/>`_. Additionally, you will also need a C++20
compiler. On Windows, this means you should install the community version of
Visual Studio, and on Linux you should install a recent version of g++ (you
will have to look for online for details regarding your particular Linux
distribution). Users on macOS likely already have a sufficient compiler.

You should start by opening a terminal (command prompt on Windows) and
navigating to the directory where you would like to store your Scarabée source
files. Once there you can run the command

.. code-block:: bash

   git clone https://github.com/scarabee-dev/scarabee.git

to download the source code into a new directory called `scarabee`, and then
run

.. code-block:: bash

   cd scarabee

to navigate into that directory. If you would like to build the development
version of Scarabée, you should checkout the development branch using

.. code-block:: bash

   git switch develop

At this point, you are ready to build Scarabée using the command

.. code-block:: bash

   pip install --user -v .

This will launch the compilation sequence, which could take several minutes to
complete.

Nuclear Data Libraries
======================

Building a Library
------------------

To use all features of Scarabée, particularly if you want to perform PWR
assembly calculations, you will need a multi-group nuclear data library (NDL).
To build an NDL for Scarabée you will need a Unix-like operating system (such as
Linux or macOS). If you are on Windows, you can use the `Windows Subsystem for
Linux (WSL) <https://learn.microsoft.com/en-us/windows/wsl/install>`_ to
completethis task.

You will also need to download the ENDF files you want to use for your library.
There are many different publicly available libraries, the most popular of which
are `ENDF <https://www.nndc.bnl.gov/endf/>`_,
`JEFF <https://www.oecd-nea.org/dbdata/jeff/>`_, and
`JENDL <https://wwwndc.jaea.go.jp/jendl/jendl.html>`_. The Scarabée source
repository contains a pre-generated script to produce a library for
`ENDF/B-VIII.0 <https://www.nndc.bnl.gov/endf-b8.0/download.html>`_. You are
certainly able to generate a library using a different evaluation should you
desire. If you do this, please consider contributing your script to the Scarabée
project ! Download the ENDF files of your choice and store them in a place where
they can be accessed by the processing script later on.

In addition to having already installed Scarabée, you also need to install the
Python packages `ENDFtk <https://github.com/njoy/ENDFtk>`_ and
`PapillonNDL <https://github.com/HunterBelanger/papillon-ndl>`_. Both must be
compiled from source.

To process the ENDF files, the
`FRENDY <https://rpg.jaea.go.jp/main/en/program_frendy/>`_ nuclear data
processing code is used. You should build the program and make sure that the
executable is available in your path.

.. Warning::

  When you use the makefile provided by FRENDY, it will produce an executable
  called ``frendy.exe``, but the scripts with Scarabée assume that it is simply
  called ``frendy``, without any extension ! Therefore, you should be sure to
  rename the executable after it has been compiled.

In the `data` folder of the source repository repository, you will find the
example script to generate a library from ENDF/B-VIII.0 files. You should use
this script as a base, and modify it to your needs. If you are going to make an
ENDF/B-VIII.0 library, you should only need to modify the first few lines in
the block at the top of the file (i.e. the location of the ENDF files, desired
temperatures, desired group structure, etc.). After this is complete, you can
run the python script from within the `data` directory, and it should begin to
process your data library.

.. Warning::

  Generating a nuclear data library is **extremely** computationally intensive.
  Depending on the number of nuclides and the number of temperatures, it could 
  take a dedicated PC up to 1 week to complete processing.

.. Tip::

   If for some reason, the script dies in the middle of processing, you do not
   need to restart from scratch once you have fixed the problem ! You can
   simply comment out the library information lines and the nuclides which were
   processed sucessfully, then re-start the script.

After the script has completed, you should have a new HDF5 formated file which
contains the entire nuclear data library and depletion chain.


Using a Library
---------------

Once you have a nuclear data library file, you should move it to a safe
location where you aren't likely to accidentally delete it ! It took a long time
to generate, so it would be a shame to lose it !

When running an assembly calculation with Scarabée, you can optionally provide
the path to the library you want to use:

.. code-block:: Python

  from scarabee import NDLibrary

  ndl = NDLibrary("/path/to/endf8_shem281.h5")

This is a convenient method to be able to quickly change libraries. However,
you might have a favorite library/group structure which you want to use all the
time. For such a case, you can set the ``SCARABEE_ND_LIBRARY`` environment
variable on your machine to be the path to your prefered library. If this
variable is set, Scarabée will use that library when loading an NDL.

.. code-block:: Python

  from scarabee import NDLibrary
  
  # Loading NDL from the SCARABEE_ND_LIBRARY environment variable path
  ndl = NDLibrary()

