import 'typeface-roboto'

import green from '@material-ui/core/colors/green'
import orange from '@material-ui/core/colors/orange'
import CssBaseline from '@material-ui/core/CssBaseline'
import { MuiThemeProvider, createTheme } from '@material-ui/core/styles'
import React from 'react'

// A theme with custom primary and secondary color.
// It's optional.
const theme = createTheme({
  typography: {
    fontFamily: 'Roboto'
  },
  palette: {
    primary: {
      light: orange[300],
      main: orange[500],
      dark: orange[700],
      contrastText: '#fff'
    },
    secondary: {
      light: green[300],
      main: green[500],
      dark: green[700],
      contrastText: '#fff'
    }
  }
})

function withRoot(Component: React.ComponentType) {
  function WithRoot(props: object) {
    // MuiThemeProvider makes the theme available down the React tree
    // thanks to React context.
    return (
      <MuiThemeProvider theme={theme}>
        {/* Reboot kickstart an elegant, consistent, and simple baseline to build upon. */}
        <CssBaseline />
        <Component {...props} />
      </MuiThemeProvider>
    )
  }

  return WithRoot
}

export default withRoot
