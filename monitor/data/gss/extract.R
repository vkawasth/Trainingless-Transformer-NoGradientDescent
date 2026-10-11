e <- new.env(); load("data/gss_all.rda", envir = e); df <- get(ls(e)[1], envir = e); rm(e); gc()
cat("dims", dim(df), "\n")
nat <- grep("^nat", names(df), value = TRUE); cat("nat columns:", nat, "\n")
want <- intersect(c("year","id","ballot","version","form","formwt","wtssall","wtssps","vpsu","vstrat","age","sex","race","degree","polviews","partyid",
  nat, "happy","health","vote12","vote16","vote20","pres12","pres16","pres20","cappun","gunlaw","grass","abany","helppoor","helpnot","helpsick","eqwlth","trust","fair","helpful","confed","conpress","conmedic","conjudge"), names(df))
sub <- df[, want]; rm(df); gc()
strip <- function(x) { x <- as.vector(unclass(x)); x }
sub[] <- lapply(sub, strip)
write.csv(sub, "gss_extract.csv", row.names = FALSE); cat("wrote", dim(sub), "\n")
