Plots pickup files from channel run.

To generate grid file add

```
    # Write grid parameters to JLD2
    save(vtkpath * "/spex_01_grid.jld2",
          Dict("N"=>N,
               "xrange"=>xrange,
               "yrange"=>yrange,
               "zrange"=>zrange,
               "periodicity"=>(true,false,false),
              )
        )
```

after

```
    cbvector = make_callbacks(
        vtkpath,
        step,
        cb_ntFrq,
        timeend,
        mpicomm,
        odesolver,
        dg,
        model,
        Q_3D,
        barotropic_dg,
        barotropicmodel,
        Q_2D,
    )
```

and add

```
using FileIO
```

after

```
using Logging, Printf, Dates
```

in top-level run script.
