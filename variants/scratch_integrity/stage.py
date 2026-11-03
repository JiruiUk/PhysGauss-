"""Pure activation and topology invariants for cold-start bindings."""


def schedule(iteration,start,interval,ramp):
    if start<15000 or interval<1 or ramp<1:raise ValueError('Invalid stable-topology activation schedule')
    elapsed=iteration-start
    return min(1.,max(0.,elapsed/ramp)),elapsed>0 and elapsed%interval==0


def legacy_iteration(iteration,start):
    if iteration<=start:raise ValueError('Geometry cannot run before activation')
    return 50000+iteration-start


def verify_binding_boundary(start,source_iteration,point_count,bound_count):
    if start<15000 or source_iteration!=start or point_count!=bound_count:
        raise ValueError('Binding belongs to a different prefix or Gaussian topology')
