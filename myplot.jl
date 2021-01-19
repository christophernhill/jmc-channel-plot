using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.BalanceLaws: vars_state, Prognostic, Auxiliary
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates: flattenednames
using ClimateMachine.Ocean.SplitExplicit01
using ClimateMachine.GenericCallbacks
using ClimateMachine.VTK
using ClimateMachine.Checkpoint

using Test
using MPI
using Random
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

## using FileIO
using JLD2
using QuickVizExample
using GLMakie

function spex01_get_checkpoints(idir="jld-inputs")
        jp(x)=joinpath(idir,x)
        bcfiles=map( jp, filter(x -> occursin("baroclinic_checkpoint"  ,x), readdir( idir ) )
                   )
        btfiles=map( jp, filter(x -> occursin("barotropic_checkpoint"  ,x), readdir( idir ) )
                   )
        gfile=map(   jp, filter(  x -> occursin("spex_01_grid",x), readdir( idir ) )
                 )
        if length(gfile) == 1
         gfile=gfile[1]
        end
 return gfile,bcfiles,btfiles
end

function spex01_lookat(
                       file_list=nothing,
                       grid_file=nothing;
                       is_baroclinic=false,
                       is_barotropic=false,
                       plot_points=nothing,
                       np_def=10
                       )

         if !is_baroclinic && !is_barotropic
           # Figure out from file names
           if any(map( x->occursin("baroclinic_checkpoint",x),file_list ))
             is_baroclinic=true
           end
           if any(map( x->occursin("barotropic_checkpoint",x),file_list ))
             is_barotropic=true
           end
         end
         # Check we are only doing 1 of barotropic and baorclinic
         if is_baroclinic && is_barotropic
           # Can't do both barotropic and baroclinic rules are same time
           println("Can't plot both barotropic and baroclinic at once")
           return
         end

         # Read and process grid file - create grids for 3d xyz (baroclinic) and 2d xy
         # (barotropic).
         jld_grid=jldopen(grid_file)
         xr,yr,zr=jld_grid["xrange"],jld_grid["yrange"],jld_grid["zrange"]
         Nel=jld_grid["N"]
         close(jld_grid)
         t3d=StackedBrickTopology( MPI.COMM_WORLD,(xr,yr,zr) )
         g3d=DiscontinuousSpectralElementGrid(t3d,
                                              FloatType=Float64,
                                              DeviceArray=ClimateMachine.array_type(),
                                              polynomialorder=Nel )
         t2d=StackedBrickTopology( MPI.COMM_WORLD,(xr,yr) )
         g2d=DiscontinuousSpectralElementGrid(t2d,
                                              FloatType=Float64,
                                              DeviceArray=ClimateMachine.array_type(),
                                             polynomialorder=Nel )
         gh3=GridHelper(g3d)
         gh2=GridHelper(g2d)
         x3,y3,z3=coordinates(g3d)
         x2,y2,z2=coordinates(g2d)

         # Create interpolation grid
         if plot_points == nothing
          xnew=range(xr[1],xr[end],length=np_def)
          ynew=range(yr[1],yr[end],length=np_def)
          znew=range(zr[1],zr[end],length=np_def)
         else
          xnew=plot_points[1]
          ynew=plot_points[2]
          znew=plot_points[3]
         end
         if is_barotropic
          znew=range(0,0,length=1)
         end

         # Load pickup data, interpolate and put into states for plotting
         npf=length(file_list)
         if is_barotropic
          fld1 = zeros(length(xnew), length(ynew), npf)
          fld2 = zeros(length(xnew), length(ynew), npf)
          fld3 = zeros(length(xnew), length(ynew), npf)
          ϕ = ScalarField(copy(x2), gh2)
          nf=1
          for fn in file_list
           jlf=jldopen(fn)
           # U[1],U[2],η
           ϕ .= view(jlf["h_Q"],:,1,:)
           fld1[:,:,nf] .= view(ϕ(xnew,ynew),:,:)
           ϕ .= view(jlf["h_Q"],:,2,:)
           fld2[:,:,nf] .= view(ϕ(xnew,ynew),:,:)
           ϕ .= view(jlf["h_Q"],:,3,:)
           fld3[:,:,nf] .= view(ϕ(xnew,ynew),:,:)
           close(jlf)
           nf=nf+1
          end
          states=[fld1, fld2, fld3]
          statenames=["U", "V", "η"]
          plot_head="Barotropic "
         end

         if is_baroclinic
          fld1 = zeros(length(xnew), length(ynew), length(znew))
          fld2 = zeros(length(xnew), length(ynew), length(znew))
          fld3 = zeros(length(xnew), length(ynew), length(znew))
          fld4 = zeros(length(xnew), length(ynew), length(znew))
          ϕ = ScalarField(copy(x3), gh3)
          fn=file_list[1]
          jlf=jldopen(fn)
          # u[1],u[2],η,θ
          ϕ .= view(jlf["h_Q"],:,1,:)
          fld1[:,:,:] .= view(ϕ(xnew,ynew,znew),:,:,:)
          ϕ .= view(jlf["h_Q"],:,2,:)
          fld2[:,:,:] .= view(ϕ(xnew,ynew,znew),:,:,:)
          ϕ .= view(jlf["h_Q"],:,3,:)
          fld3[:,:,:] .= view(ϕ(xnew,ynew,znew),:,:,:)
          ϕ .= view(jlf["h_Q"],:,4,:)
          fld4[:,:,:] .= view(ϕ(xnew,ynew,znew),:,:,:)
          close(jlf)
          states=[fld1, fld2, fld3, fld4 ]
          plot_tim=@sprintf("%g",jlf["t"])
          statenames=["u, t="*plot_tim, "v, t="*plot_tim, "η, t="*plot_tim, "θ, t="*plot_tim ]
          plot_head="Baroclinic "
         end

         volumeslice(states, statenames = statenames, title=plot_head)

         return
end

gfile,bcfiles,btfiles=spex01_get_checkpoints("/Users/chrishill/projects/QuickVizExample.jl/jld-inputs")
spex01_lookat(btfiles,gfile;np_def=100)
# spex01_lookat([bcfiles[30]],gfile;np_def=100)
