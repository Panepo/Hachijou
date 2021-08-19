var addon = require('bindings')('addon.node')

console.log('This should be 134629:', addon.factorializeFib(30))
