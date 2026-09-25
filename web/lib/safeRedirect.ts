/** Only same-origin relative paths: never bounce a user to an attacker-supplied URL after signing in. */
export function safeRedirect(target: string | undefined): string {
  return target && target.startsWith('/') && !target.startsWith('//') && !target.startsWith('/login')
    ? target
    : '/projects'
}
